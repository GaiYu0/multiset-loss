import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from utilities import onehot

'''
def loss(data, labels):
  labels = th.sum(lables.detach(), 1)
  labels = th.squeeze(labels)
  data = th.sum(data, 1)
  data = th.squeeze(data)
  loss = F.kl_div(labels, data)
  N, T, C = data.size()
  chunks = th.chunk(data, T, 1)
'''

def loss(data, labels):
  # data (N, T, C)
  # labels (N, T, C) no duplication

  N, T, C = data.size()
  chunks = th.chunk(data, T, 1)
  mask = th.sum(labels.data, 1)
  mask = th.squeeze(mask)
  loss = 0

  for index, chunk in enumerate(chunks):
    chunk = th.squeeze(chunk)
    chunk = F.log_softmax(chunk)
    loss += th.sum(Variable(mask) * chunk, 1) / (T - index)
    _, p = th.max(chunk, 1)
    onehot_p = onehot(p.data, 10)
    mask = mask - onehot_p
    mask = th.max(th.zeros(mask.size()).cuda(), mask)
# import pdb; pdb.set_trace()
  loss = -th.mean(loss)
  return loss
