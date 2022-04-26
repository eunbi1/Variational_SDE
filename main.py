
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


from sde_lib import VPSDE
from models import ddpm
from models import utils as mutils
import datasets, losses

#SDE 설정
sigma_min = 0.01
sigma_max = 50
num_scales = 1000
beta_min = 0.1
beta_max = 20.
num_scales = 1000

sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
sampling_eps = 1e-3

batch_size = 64
random_seed = 0

sigmas = mutils.get_sigmas(sigma_max,sigma_min,num_scales)
scaler = datasets.get_data_scaler(centered = True)
inverse_scaler = datasets.get_data_inverse_scaler(centered = True)

score_model = ddpm.DDPM()
score_model = torch.nn.DataParallel(score_model)
optimizer = losses.get_optimizer(score_model.parameters())


# dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#총 50000개의 image가 있다. channel =3, x,y =32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def image_grid(x):
  size = 32
  channels = 3
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()
#images, labels = next(iter(trainloader))
#show_samples(images)

"""
def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std
"""

def loss_fn(model, x, marginal_prob, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob(x, random_t)[1]
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

n_epochs =   50#@param {'type':'integer'}

lr=1e-4 #@param {'type':'number'}


#sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)




n_epochs = 1

#tqdm_epoch = tqdm.notebook.trange(n_epochs)
for epoch in range(n_epochs):
  avg_loss = 0.
  num_items = 0
  print('돌아가?')
  for x, y in trainloader:
    loss = loss_fn(score_model, x, sde.marginal_prob)
    print(f'loss : {loss}, epoch{epoch}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  print('Average Loss: {:5f}'.format(avg_loss / num_items))

img_size = 32
channels = 3
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor
#@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector
#@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow)

x, n = sampling_fn(score_model)
show_samples(x)