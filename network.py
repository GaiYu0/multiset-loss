from pdb import set_trace as st

import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self._conv0 = nn.Conv2d(1, 6, 5, 1, 2)
    self._conv1 = nn.Conv2d(6, 16, 5, 1, 2)
    self._linear0 = nn.Linear(16 * 7 * 7, 84)
    self._linear1 = nn.Linear(84, 10)

  def forward(self, data):
    size = data.size()
    if len(size) != 4:
      data = data.contiguous()
      data = data.view(size[0], 1, 28, 28)
    data = F.relu(self._conv0(data))
    data = F.max_pool2d(data, 2, 2)
    data = F.relu(self._conv1(data))
    data = F.max_pool2d(data, 2, 2)
    data = data.contiguous().view(size[0], -1)
    data = F.relu(self._linear0(data))
    data = self._linear1(data)
    return data

class RNN(nn.Module):
  def __init__(self, n_units, n_classes, cnn_path, cuda):
    super(RNN, self).__init__()
    self._cnn = CNN()
    if n_units > 0:
      self._fh = nn.Linear(10, n_units)
      self._hh = nn.Linear(n_units, n_units)
      self._classifier = nn.Linear(n_units, n_classes)
    self._n_units = n_units
    self._contextualize = lambda t: t.cuda() if cuda else t

    if cnn_path:
      self._pretrained_cnn = True
      self._cnn.load_state_dict(th.load(cnn_path))
    else:
      self._pretrained_cnn = False

  def forward(self, data):
    N, T, _ = data.size()
    if self._n_units > 0:
      h = th.zeros(N, self._n_units)
      h = self._contextualize(h)
      h = Variable(h)
    chunks = th.chunk(data, T, 1)
    chunks = map(th.squeeze, chunks)
    pre = []
    for ch in chunks:
      f = self._cnn(ch)
      if self._pretrained_cnn:
        f = f.detach()
      if self._n_units > 0:
        h = F.tanh(self._fh(f) + self._hh(h))
        c = self._classifier(h)
      else:
        c = f
      pre.append(c)
    pre = tuple(th.unsqueeze(p, 1) for p in pre)
    pre = th.cat(pre, 1)
    return pre
