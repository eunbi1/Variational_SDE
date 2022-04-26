
def get_data_scaler(centered = True):
  """Data normalizer. Assume data are always in [0, 1]."""
  if centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(centered = True):
  """Inverse data normalizer."""
  if centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x