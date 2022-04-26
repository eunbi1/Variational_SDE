import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

""" <weight initilization tool>
    variance_scaling 
    default_init(scale=1.
    
    <Layer 종류 >
     NIN
     RefineBlock
     ResidualBlock
     ResnetBlockDDPM
     Upsample
     Downsample
     ddpm_conv3x3
     AttnBlcok
     
     <function 종류>
     def _einsum(a, b, c, x, y):
     contract_inner(x, y)
     get_timestep_embedding(timesteps, embedding_dim, max_positions=10000)

"""


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


# 특정 함수의 parameter를 결정짓고 싶을 때 함수 내부에 함수를 저장하고 return으로 함수를 받음
# size만큼의 initialization왼 tensor를 받게됨

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  # string.ascii_lowercase = 'abcdefgijklmnopqrstuvwxyz'
  x_chars = list(string.ascii_lowercase[:len(x.shape)]) # len(x.shape)=3 x_chars = [a,b,c]
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)]) # len(y.shape)=2 y_chars =[d,e]
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

#network in network : chammel에 대해서만 FC를 적용한 것과 같은 효과이다.
#같은 문자인 경우 Batch or summation
class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1) #(out_ch, in_ch, x,y) -> (out_ch, x,y,in_ch)
    y = contract_inner(x, self.W) + self.b # (a,b,c,d), (d,e) ->(a,b,c,e) in_ch에 대해서 c->e 변경
    return y.permute(0, 3, 1, 2) #( out_ch,e,x,y)

class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=0.)

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)# 이게 channel-wise FC이다.
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W)) # key는 pixel 단위로 간주해서 총 H*W개 있음.
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)# channel-wise self attention을 왜해?
    h = self.NIN_3(h)
    return x + h


class RefineBlock(nn.Module):
  def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
    super().__init__()

    assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
    self.n_blocks = n_blocks = len(in_planes)

    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

    self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

    if not start:
      self.msf = MSFBlock(in_planes, features)

    self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

  def forward(self, xs, output_shape):
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i])
      hs.append(h)

    if self.n_blocks > 1:
      h = self.msf(hs, output_shape)
    else:
      h = hs[0]

    h = self.crp(h)
    h = self.output_convs(h)

    return h

class ResidualBlock(nn.Module):
  def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
               normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
    super().__init__()
    self.non_linearity = act
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.resample = resample
    self.normalization = normalization
    if resample == 'down':
      if dilation > 1:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
      else:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
        conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

    elif resample is None:
      if dilation > 1:
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
      else:
        # conv_shortcut = nn.Conv2d ### Something wierd here.
        conv_shortcut = partial(ncsn_conv1x1)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim)
    else:
      raise Exception('invalid resample value')

    if output_dim != input_dim or resample is not None:
      self.shortcut = conv_shortcut(input_dim, output_dim)

    self.normalize1 = normalization(input_dim)

  def forward(self, x):
    output = self.normalize1(x)
    output = self.non_linearity(output)
    output = self.conv1(output)
    output = self.normalize2(output)
    output = self.non_linearity(output)
    output = self.conv2(output)

    if self.output_dim == self.input_dim and self.resample is None:
      shortcut = x
    else:
      shortcut = self.shortcut(x)

    return shortcut + output

class ResnetBlockDDPM(nn.Module):
  """The ResNet Blocks used in DDPM."""
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
    super().__init__()
    if out_ch is None:
      out_ch = in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
    self.act = act
    self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.) # 왜 여긴 init_scale = 0.로 설정한거야?
    if in_ch != out_ch:
      if conv_shortcut: # 이게 뭐지?
        self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)
    self.out_ch = out_ch
    self.in_ch = in_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    B, C, H, W = x.shape
    assert C == self.in_ch
    out_ch = self.out_ch if self.out_ch else self.in_ch
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if C != out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x) # convolution으로 x의 channel dimension을 변경
      else:
        x = self.NIN_0(x) # channel dimension을 contraction을 통해서 변경
    return x + h

class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
      super().__init__()
      if with_conv:
        self.Conv_0 = ddpm_conv3x3(channels, channels)
      self.with_conv = with_conv

    def forward(self, x):
      B, C, H, W = x.shape
      h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
      if self.with_conv:
        h = self.Conv_0(h)
      return h

class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
      super().__init__()
      if with_conv:
        self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
      self.with_conv = with_conv

    def forward(self, x):
      B, C, H, W = x.shape
      # Emulate 'SAME' padding
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1)) # [N,C,h+1,w+1] right에다가 padding넣음
        x = self.Conv_0(x) # [ N, C. h/2, w/2] (h-k+s+p)/s 한쪽만 padding했으므로
      else:
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

      assert x.shape == (B, C, H // 2, W // 2)
      return x

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

def get_coordinate_embedding(z, n_min=7, n_max=8, ch_axis = 1 ):
  range_n = range(n_min, n_max+1)
  fourier_features = torch.cat([ torch.cat( [torch.sin(z*2**n *torch.pi), torch.cos(z*2**n*torch.pi)  ], dim=ch_axis ) for n in range_n],dim=ch_axis)
  z = torch.cat([z, fourier_features], dim =1 )
  return z

def monotone_net(t):
  l1 = nn.Linear(1, 1)
  l2 = nn.Linear(1,1024)
  l3 = nn.Linear(1024,1)
  act = torch.sigmoid
  return l1(t)+l3(act(l2(l1(t))))
