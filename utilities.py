from pdb import set_trace as st

import torch as th
import torch.nn.functional as F

def onehot(labels, D, cuda=True):
  # labels (N, 1)
  labels = labels.cpu()
  N = labels.size()[0]
  result = th.zeros(N, D)
  result.scatter_(1, labels, 1)
  if cuda:
    result = result.cuda()
  return result

def onehot_sequence(labels, D, cuda=True):
  # labels (N, T)
  N, T = labels.size()
  onehot_labels = []
  for l in th.chunk(labels, T, 1):
    onehot_label = onehot(l, D, cuda)
    onehot_label = th.unsqueeze(onehot_label, 1)
    onehot_labels.append(onehot_label)
  onehot_labels = th.cat(onehot_labels, 1)
  return onehot_labels

def n_matches(data, labels):
  """
  Parameters
  ----------
  data: (N, T, C)
  labels: (N, T, C)
  """
  _, data = th.max(data, 2)
  data = th.squeeze(data, 2)
  _, labels = th.max(labels, 2)
  labels = th.squeeze(labels, 2)
  n = th.sum(th.prod(data == labels, 1))
  n = n.data[0]
  return n

def jsd(p, q):
  """ Jensen-Shannon divergence.
  p (N, D)
  q (N, D)
  """
  m = p + q / 2
  div = F.kl_div(m, p) + F.kl_div(m, q)
  return div
