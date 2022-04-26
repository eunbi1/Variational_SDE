
import torch
import torch.nn as nn
import functools

from models import utils, layers, normalization

# parameters
# training.continuous = True
# training.reduce_mean = True

# sampling
# sampling.method = 'pc'
# sampling.predictor = 'euler_maruyama'
# sampling.corrector = 'none'

# data
# data.centered = True
# data image_size =32

# model
# model.scale_by_sigma = False
# model.ema_rate = 0.9999
# model.normalization = 'GroupNorm'
# model.nonlinearity = 'swish'
# model.nf = 128
# model.ch_mult = (1, 2, 2, 2)


# model.resamp_with_conv = True
# model.conditional = True

#SDE 설정
sigma_min = 0.01
sigma_max = 50
num_scales = 1000
beta_min = 0.1
beta_max = 20.

# 각 block의 의미 파악하기
RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
#AttnBlock
num_res_blocks = 2
attn_resolutions = (16,)

#normalization
norm = nn.GroupNorm

#Xavier initialization
default_initializer = layers.default_init

class DDPM(nn.Module):
  def __init__(self, sigma_max = 50,sigma_min = 0.01 ,num_scales = 1000,
               nf=128, ch_mult = (1, 2, 2, 2), num_res_blocks = 2, attn_resolutions = (16,),
               dropout = 0.2, resamp_with_conv = True, conditional = True, channels=3, centered = True):

    super().__init__()
    self.act = act = nn.SiLU() # swish activator
    #def get_sigmas(sigma_max, sigma_min, num_scales):
    # register_buffer가 뭐지?
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(sigma_max,sigma_min,num_scales)))
    # update하지 않는 parameter 등록을 위한 것

    self.nf = nf = nf  # 무슨 의미? number of features 3->128->
    ch_mult =  ch_mult # chammel multiplication
    self.num_res_blocks = num_res_blocks
    self.attn_resolutions = attn_resolutions
    dropout = dropout
    resamp_with_conv = resamp_with_conv # 이건 무슨 기능이야?
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [32 // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    #channel-wise self-attention block

    self.conditional = conditional = conditional # 이게 무슨 의미
    # CNN layer 존재-> CNN ; convolution Resnet image high resolution, layer 많이.
    # 깊게 쌓으면 gradient에 문제 -> 해결하려고 도입 ; resnetblock -> 많은 층을 쌓더라도 학습이 안정.
    # high resolution 처리 보편적으로 사용.
    # 64x64, 128x128
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)

    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)] # 4*nf?
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = centered
    channels = channels # data number channel은 color channel인 3이다.

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf #128 <-(feature extraction) channel =3
    # ch_mult = [1,2,2,2]
    #self.num_resolutions = num_resolutions = len(ch_mult)
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      # num_res_blocks =2
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        # 128->128-(Downsample)>>-256->256-(Downsample)>>-256->256-(Downsample)>>256->256
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        # 특정 resolution에 왜 attention? (질문)
        if all_resolutions[i_level] in attn_resolutions: # [32,16,8,4] spatial dimension
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1] # 256
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1): # 왜 이거 하나 더 늘었어?
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0 :
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = False # 이게 뭐지?

  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels # 여기에 감마 t
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h
