import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Do one of:

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=10000):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self,  channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)

  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))


    return h

import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, model_type='ddpm', in_channels=3,out_ch=3, ch=128, ch_mult=[1,2,2,2],
                 num_res_blocks=2, attn_resolutions=[16,], dropout=0.1, 
                 ema_rate=0.9999, ema=True, resamp_with_conv=True,
                 ckpt_dir='/content/cifar-10-batches-py'):
        super().__init__()
        resolution = 28
        ch_mult = tuple(ch_mult)
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

"""# Define SDE 

"""

if torch.cuda.is_available(): 
  device = 'cuda' 
else: 
  device= 'cpu' 

print(device )
class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, T=1., device =device):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha = alpha
        self.T = T

    def beta(self,t):
      return (self.beta_1 - self.beta_0)*t + self.beta_0 
      
    def marginal_log_mean_coeff(self, t):
        t = torch.tensor(t, device = device)
        log_alpha_t = - 1/(2*alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1/alpha * t * self.beta_0
        return log_alpha_t

    def diffusion_coeff(self, t):
      return torch.exp(self.marginal_log_mean_coeff(t))


    def marginal_std(self, t):
        t = torch.tensor(t, device = device)
        sigma = torch.pow(1. - torch.exp(self.alpha*self.marginal_log_mean_coeff(t)), 1/self.alpha)  
        return sigma

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha*log_mean_coeff), 1/self.alpha))
        return log_mean_coeff - log_sigma

"""# score function """

import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt 
from functools import partial

class score_fn():

  def __init__(self, alpha=1.9,sigma1=1, sigma2=1, t0=5, Fs=15):
    self.alpha = alpha
    self.Fs = Fs 
    self.sigma_1=sigma1
    self.sigma_2=sigma2
    self.t=t=np.arange(-t0,t0,1./Fs)
    self.f = f = np.linspace(-Fs/2, Fs/2, len(t))
    self.scores = self.score_function(self.g1,self.g2, t, f)
    

  def score_function(self, g1,g2, t,f):
    approx1 = np.fft.fftshift(np.fft.fft(g1(t)) * np.exp(2j*np.pi*f*t0) * 1/Fs)
    approx2 = np.fft.fftshift(np.fft.fft(g2(t)) * np.exp(2j*np.pi*f*t0) * 1/Fs)
    approx_score = np.divide(approx2, approx1)
    return np.divide(approx2, approx1)

  
  def g1(self, x): 
    return np.exp(-1/2*(2*np.pi*x*self.sigma_1)**2)*np.exp(-np.power(2*np.pi*np.abs(x*self.sigma_2), self.alpha))

  def g2(self, x):
    return (-2j*np.pi*x)*np.exp(-1/2*(2*np.pi*x*self.sigma_1)**2)*np.exp(-np.power(2*np.pi*np.abs(x*self.sigma_2), self.alpha))

  def point_evaluation(self, x):
    # x : number 
    # output: number
    if x <= self.Fs/2:
      k = np.argmin(np.abs(self.f-x),axis=0)
      return np.real(self.scores[k])
    else : 
      print(x)
      Fs = 2*x+1 
      self.f= f= np.linspace(-Fs/2, Fs/2, len(self.t))
      self.scores = self.score_function(self.g1,self.g2, self.t, f)
      k = np.argmin(np.abs(self.f-x))
      return np.real(self.scores[k])

  def evaluation(self,x):

    result = np.zeros(x.shape, dtype=complex)
    xxx = np.nditer(x, flags=['multi_index'])

    for element in xxx:
        i = xxx.multi_index 
        result[i] = self.point_evaluation(element)
      
    return result

t0=5
Fs = 15
t=np.arange(-t0,t0,1./Fs)

f = np.linspace(-Fs/2, Fs/2, len(t))

score = score_fn(alpha=1.5)


exact_score = score.scores




plt.subplot(132)
plt.plot(f, np.real(exact_score), 'r-', lw=2)

    
plt.subplot(133)

plt.plot(f, np.imag(exact_score), 'b')


g = np.random.randn(3,2,2)
score.evaluation(g).shape

"""# Loss function

"""

import torch
import copy
import time 
from scipy.special import gamma

def get_continuous_time(index):
        return (index+1) / 1000

# model은 [0,1]이지만 학습할 때는 Integer로 들어간다.
def get_discrete_time(t):
    return 1000. * torch.max(t - 1. / 1000, torch.zeros_like(t).to(t))

def get_beta_schedule(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    return np.linspace( beta_start, beta_end, num_timesteps, dtype=np.float64)

def gamma_fn(x):
    return torch.tensor(gamma(x))

def loss_fn(model,x0: torch.Tensor,
              t: torch.LongTensor,
              e_L: np.array,
              b: torch.Tensor,
              alpha , keepdim = False,
              approximation=True):

    aa = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  
    sigma = (torch.pow(1-aa,1/alpha)).to(device)
    sigma=sigma.view(-1,1,1,1)
    t0 = time.time()
    score = score_fn(alpha)
    
    sigma_score = score.evaluation(e_L)
    t1=time.time()
    e_L = torch.Tensor(e_L).to(device)
    sigma_score = torch.Tensor(sigma_score).to(device )
    
    x_t = torch.pow(aa, 1/alpha)*x0+ e_L * sigma

  
    if approximation: 
      sigma_score = -  gamma_fn(3/alpha)/gamma_fn(1/alpha)*e_L
    output = model(x_t, t.float())
    weight= (sigma*output - sigma_score)


    return (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)

"""# training 


"""

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import tqdm
from scipy.stats import levy_stable
import matplotlib.pyplot as plt 
import time
if torch.cuda.is_available(): 
  device = 'cuda' 
else: 
  device= 'cpu' 


#@title Training score model

score_model = torch.nn.DataParallel(ScoreNet())
score_model = score_model.to(device)
# The time input into the score model is in [0,1]. 

#Forward process setting 
alpha=2 #@param {'type':'number'}
num_timesteps=1000 #@param {'type':'number'}

beta_start=0.0001
beta_end=0.02,
betas= get_beta_schedule(beta_start=beta_start, beta_end=beta_end, num_timesteps=num_timesteps)
b= torch.from_numpy(betas).float().to(device)

n_epochs =   5#@param {'type':'integer'}
batch_size =  64#@param {'type':'integer'}
lr=1e-4#@param {'type':'number'}

#dataset setting 
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.\
    trange(n_epochs)

L = []
counter =0 

t_0 =time.time()
for epoch in tqdm_epoch:
  counter+=1
  avg_loss = 0.
  num_items = 0
  for x, y in data_loader:
    n = x.size(0)
    x = x.to(device)    
    e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=x.shape)
    t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
    loss = loss_fn(score_model,x, t, e_L,b, alpha)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  
  L.append(avg_loss/num_items)  


  if counter%1 ==0:
    print(counter, 'th', avg_loss/num_items)
  
  # Print the averaged training loss so far.
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), 'ckpt.pth')
t_1 = time.time()
print('Total running time is', t_1-t_0)
plt.plot(np.arange(n_epochs), L)

beta_start=0.0001
beta_end=0.02,
num_timesteps=1000
betas= get_beta_schedule(beta_start=beta_start, beta_end=beta_end, num_timesteps=num_timesteps)
b= torch.from_numpy(betas).float().to(device)
x0,y = next(iter(data_loader))
x0 = x0.to(device)

print(b.shape)
print(x.shape)
t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
print(t.shape)

aa = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(device)
sigma = (torch.pow(1-aa,1/alpha)**2).view(-1,1,1,1).to(device)
print('aa', aa.shape)
print('simg', sigma.shape)



s = torch.linspace(1,0.001,10)
t = s[1:]
s= s[:-1]
print(t,s)

# ddim_score_update(score_model, x0,s,t)

"""# Sampling

"""

def get_discrete_time(t):
        return 1000. * torch.max(t - 1. / total_N, torch.zeros_like(t).to(t))

        # -1/N인인 이유는는 시간이이 t=1/N일일 때때 x0 번째째 chain이이 되도록록 설정하기 위함함.

def gamma_func(x):
    return torch.tensor(gamma(x))

def get_discrete_time(t):
        return 1000. * torch.max(t - 1. /1000, torch.zeros_like(t).to(t))

        # -1/N인인 이유는는 시간이이 t=1/N일일 때때 x0 번째째 chain이이 되도록록 설정하기 위함함. 


def ddim_score_update(model, sde, x_s, s, t,  return_noise=False):

        score_s = model(x_s, get_discrete_time(s))
        
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)

        
        sigma_s= sde.marginal_std(s)

        time_step = torch.abs(t - s)
        e_L = levy_stable.rvs(alpha=sde.alpha, beta=0, loc=0, scale=1, size=x_s.shape)
        e_L = torch.Tensor(e_L).to(device)
        x_coeff = 2-torch.pow(1-sde.beta(time_step), 1/sde.alpha)
        noise_coeff = torch.pow(sde.beta(time_step),1/sde.alpha)

        if alpha==2:
          score_coeff = 2*noise_coeff**2
    
        else:
          score_coeff = 2*gamma_func(sde.alpha+1)/torch.pi*np.sin(torch.pi/2*(2-sde.alpha))*torch.pow(noise_coeff,2)/(2-sde.alpha)*torch.pow(time_step,1-2/sde.alpha)[:,None,None,None]
        noise_coeff = sigma_s*torch.pow(time_step,1/sde.alpha)


        x_t =  x_coeff[:,None,None,None] * x_s + score_coeff[:,None,None,None] * score_s+ sigma_s[:,None,None,None]*e_L

        if return_noise:
            return x_t, score_s
        else:
            return x_t

def ddim_score_update2(model, sde, x_s, s, t, h=0.006, return_noise=False):
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
        lambda_s= sde.marginal_lambda(s)
        lambda_t = sde.marginal_lambda(t)
        sigma_s = sde.marginal_std(s)
        sigma_t = sde.marginal_std(t)

        score_s = model(x_s, get_discrete_time(s))
        h_t = lambda_t - lambda_s
        
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)

        
        sigma_s= sde.marginal_std(s)

        time_step = torch.abs(t - s)
        e_L = levy_stable.rvs(alpha=sde.alpha, beta=0, loc=0, scale=1, size=x_s.shape)
        e_L = torch.Tensor(e_L).to(device)
        x_coeff = 2-torch.pow(1-sde.beta(time_step), 1/sde.alpha)
        noise_coeff = torch.pow(sde.beta(time_step),1/sde.alpha)

        score_coeff = 2*sigma_t * torch.pow(sigma_s, alpha-1) * alpha * torch.expm1(h_t) \
                     * gamma_func(alpha-1) / torch.pow(gamma_func(alpha/2),2) / np.power(h, alpha-2)

        x_t =  x_coeff[:,None,None,None] * x_s + score_coeff[:,None,None,None] * score_s+ sigma_s[:,None,None,None]*e_L

        if return_noise:
            return x_t, score_s
        else:
            return x_t

def dpm_score_update(model, x_s, s, t, alpha, h=0.006, return_noise=False):
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
        lambda_s= sde.marginal_lambda(s)
        lambda_t = sde.marginal_lambda(t)
        sigma_s = sde.marginal_std(s)
        sigma_t = sde.marginal_std(t)

        score_s = model(x_s, get_discrete_time(s))


        h_t = lambda_t - lambda_s

        x_coeff = torch.exp(log_alpha_t - log_alpha_s)
        

        score_coeff = sigma_t * torch.pow(sigma_s, alpha-1) * alpha * torch.expm1(h_t) \
                     * gamma_func(alpha-1) / torch.pow(gamma_func(alpha/2),2) / np.power(h, alpha-2)

        x_t =  x_coeff[:,None,None,None] * x_s + score_coeff[:,None,None,None] * score_s


        return x_t


def pc_sampler(score_model, 
               sde,
               alpha,
               batch_size, 
               num_steps,  
               LM_steps=200000,              
               device=device,
               eps=1e-3,
               Predictor=True,
               Corrector=True):
  
  t = torch.ones(batch_size, device=device)
  sigma=sde.marginal_std(t)
  print('dho dksEj')

  e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=(batch_size,1,28,28))
  e_L = torch.Tensor(e_L).to(device)
  print('du긴')
  init_x = e_L.clone().detach().requires_grad_(True)
  print(init_x)
  
  time_steps = np.linspace(1., eps, num_steps) # (t_{N-1}, t_{N-2}, .... t_0)
  step_size = time_steps[0] - time_steps[1]
  
  x_s = init_x*sigma[:,None,None,None]
  i = 0 
  alpha = sde.alpha
  batch_time_step_s = torch.ones(batch_size, device=device) *time_steps[0]


  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps[1:]):
      
      batch_time_step_t = torch.ones(batch_size, device=device) * time_step
      print('batch_s', batch_time_step_s, '\n', 'batch_s', batch_time_step_t)

      if Corrector :  
        for j in range(LM_steps):
          grad = score_model(x_s, get_discrete_time(batch_time_step_t))
      
          e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=(batch_size,1,28,28))
          e_L = torch.Tensor(e_L.clone().detach() ).to(device)
  
          x_s = x_s + step_size*gamma_func(sde.alpha-1)/(gamma_func(sde.alpha/2)**2)*grad  + np.power(step_size, 1/sde.alpha) * e_L

      # Predictor step (Euler-Maruyama)
      if Predictor :
        
        x_s =ddim_score_update(score_model, sde, x_s, batch_time_step_s, batch_time_step_t)

      batch_time_step_s = batch_time_step_t
      
  x_t = x_s  
  return x_t

def pc_sampler2(score_model, 
               sde,
               alpha,
               batch_size, 
               num_steps,  
               LM_steps=200000,              
               device=device,
               eps=1e-3,
               Predictor=True,
               Corrector=True):
  
  t = torch.ones(batch_size, device=device)
  sigma=sde.marginal_std(t)
  print('dho dksEj')

  e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=(batch_size,1,28,28))
  e_L = torch.Tensor(e_L).to(device)
  print('du긴')
  init_x = e_L.clone().detach().requires_grad_(True)
  print(init_x)
  
  time_steps = np.linspace(1., eps, num_steps) # (t_{N-1}, t_{N-2}, .... t_0)
  step_size = time_steps[0] - time_steps[1]
  
  x_s = init_x*sigma[:,None,None,None]
  i = 0 
  alpha = sde.alpha
  batch_time_step_s = torch.ones(batch_size, device=device) *time_steps[0]


  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps[1:]):
      
      batch_time_step_t = torch.ones(batch_size, device=device) * time_step
      print('batch_s', batch_time_step_s, '\n', 'batch_s', batch_time_step_t)

      if Corrector :  
        for j in range(LM_steps):
          grad = score_model(x_s, get_discrete_time(batch_time_step_t))
      
          e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=(batch_size,1,28,28))
          e_L = torch.Tensor(e_L.clone().detach() ).to(device)
  
          x_s = x_s + step_size*gamma_func(sde.alpha-1)/(gamma_func(sde.alpha/2)**2)*grad  + np.power(step_size, 1/sde.alpha) * e_L

      # Predictor step (Euler-Maruyama)
      if Predictor :
        
        x_s =ddim_score_update(score_model, sde, x_s, batch_time_step_s, batch_time_step_t)

      batch_time_step_s = batch_time_step_t
      
  x_t = x_s  
  return x_t
def dpm_sampler(score_model, 
               sde,
               alpha,
               batch_size, 
               num_steps,               
               device=device,
               eps=1e-3):
  
  t = torch.ones(batch_size, device=device)
  sigma=sde.marginal_std(t)


  e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=(batch_size,1,28,28))
  e_L = torch.Tensor(e_L).to(device)
  init_x = e_L
  time_steps = np.linspace(1., eps, num_steps) # (t_{N-1}, t_{N-2}, .... t_0)
  x_s = init_x*sigma[:,None,None,None]
  i = 0 
  alpha = sde.alpha
  batch_time_step_s = torch.ones(batch_size, device=device) *time_steps[0]

  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps[1:]):
      batch_time_step_t = torch.ones(batch_size, device=device) * time_step
      x_t =dpm_score_update(score_model, x_s, batch_time_step_s, batch_time_step_t, alpha)

      batch_time_step_s = batch_time_step_t
    
    # The last step does not include any noise
    return x_t

from torchvision.utils import make_grid

## Load the pre-trained checkpoint from disk.



#@title Sampling 

num_steps =  4#@param {'type': 'integer'}
alpha=2#@param {'type' : 'number'}
batch_size = 64 #@param {'type': 'integer'}
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)


#model, sde definition
sample_batch_size = 64 

sde =VPSDE(alpha=alpha)

sampler ="dpm_sampler"#@param ['dpm_sampler', 'pc_sampler', 'pc_sampler2']{'type' : 'string'}
## Generate samples using the specified sampler.
samples = dpm_sampler(score_model, 
                  sde,
                  alpha=alpha,
                  batch_size=batch_size,
                  num_steps=num_steps,
                  device=device)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)
# %matplotlib inline
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()


x,y = next(iter(data_loader))
sample_grid = make_grid(x, nrow=int(np.sqrt(sample_batch_size)))
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()