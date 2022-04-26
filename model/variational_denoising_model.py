
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

#Architecture
ResnetBlockDDPM = layers.ResnetBlockDDPM
conv3x3 = layers.ddpm_conv3x3
#AttnBlock
num_res_blocks = 32
"""
<Parameter>
1) nf = 128
2) num_res_blocks = 32
3) Dropout = 0.1
"""
#normalization
norm = nn.GroupNorm

#Xavier initialization
default_initializer = layers.default_init



class Variational_denosing_model(nn.Module):
  def __init__(self, sigma_max = 50, sigma_min = 0.01, num_scales = 1000,
               nf=128, num_res_blocks = 32,
               dropout = 0.1, resamp_with_conv = True, conditional = True, channels=3, centered = True,
               n_min = 7, n_max = 8):

    super().__init__()
    self.act = act = nn.SiLU() # swish activator
    #def get_sigmas(sigma_max, sigma_min, num_scales):
    # register_buffer를 지우고 neural net로 변경?
    sigmas = torch.tensor(utils.get_sigmas(sigma_max,sigma_min,num_scales))
    # update하지 않는 parameter 등록을 위한 것

    self.nf = nf = nf  # 무슨 의미? number of features 3->128->
    self.num_res_blocks = num_res_blocks
    dropout = dropout
    resamp_with_conv = resamp_with_conv # 이건 무슨 기능이야?
    self.channels = channels = channels

    # spatial embedding을 위해 추가한 지점
    self.n_max = n_max = n_max
    self.n_min = n_min = n_min
    self.channels_emd = channels_emd = self.channels + ( n_max - n_min +1 )*channels*2 # spatial embedding

    AttnBlock = functools.partial(layers.AttnBlock)
    # class를 함수로 변경? 뭐하는거지?
    #channel-wise self-attention block

    self.conditional = conditional = conditional # 이게 무슨 의미
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
    # Forward direction
    modules.append(conv3x3(channels_emd, nf))
    hs_c = [nf]
    in_ch = nf #128 <-(feature extraction) channels_emd =3 + 3*2*2
    for i_block in range(num_res_blocks):
        modules.append(ResnetBlock(in_ch= nf , out_ch= nf))
        hs_c.append(in_ch)

    modules.append(ResnetBlock(in_ch= nf ))
    modules.append(AttnBlock(channels= nf ))
    modules.append(ResnetBlock(in_ch= nf))

    # Reverse direction
    for i_block in range(num_res_blocks):
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=nf ))

    assert  len(hs_c)==1
    modules.append(nn.GroupNorm(num_channels=nf, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = False # 이게 뭐지?

  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels # 여기에 monotone neural network \gamma_t를 넣어야함.
      timesteps = layers.monotone_net(timesteps)
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
    h = layers.get_coordinate_embedding(h, n_min = self.n_min, n_max = self.n_max )

    # Forward direction
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        hs.append(h)
    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Reverse direction
    for i_level in reversed(range(self.num_res_blocks)):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

    assert  len(hs)==1
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

f = Variational_denosing_model()
x = torch.ones((2,3,4,4))
labels = torch.randn(1)
print('final', f(x, labels ).shape)