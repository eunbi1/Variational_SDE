import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE


def get_optimizer(params, optimizer='Adam', lr=2e-4, beta1=0.9,
                  eps=1e-8, weight_decay =0 ):
  if optimizer == 'Adam':
    optimizer = optim.Adam(params=params, lr=lr , betas=(beta1, 0.999), eps=eps, weight_decay=weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {optimizer} not supported yet!')

  return optimizer

def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn

