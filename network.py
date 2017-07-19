import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, n_classes=10):
    super(Network, self).__init__()
    self._n_classes = n_classes

    self._linear0 = nn.Linear(784, 1024)
    self._linear1 = nn.Linear(1024, 1024)
    self._linear2 = nn.Linear(1024, 1024)
    self._classifier = nn.Linear(1024, 10)

  def forward(self, data):
    N, T, _ = data.size()
    p = []
    for d in th.chunk(data, T, 1):
      d = th.squeeze(d)
      d = F.relu(self._linear0(d))
      d = F.relu(self._linear1(d))
      d = F.relu(self._linear2(d))
      d = self._classifier(d)
      d = th.unsqueeze(d, 1)
      p.append(d)
    p = th.cat(p, 1)
    return p

class CNN(nn.Module):
  def __init__(self, n_features=84):
    super(CNN, self).__init__()
    self._conv0 = nn.Conv2d(1, 6, 5, 1, 2)
    self._conv1 = nn.Conv2d(6, 16, 5, 1, 2)
    self._linear0 = nn.Linear(16 * 7 * 7, n_features)

  def forward(self, data):
    N = data.size()[0]
    data = F.relu(self._conv0(data))
    data = F.max_pool2d(data, 2, 2)
    data = F.relu(self._conv1(data))
    data = F.max_pool2d(data, 2, 2)
    data = data.contiguous().view(N, -1)
    data = F.relu(self._linear0(data))
    return data

class RNN(nn.Module):
  def __init__(self, n_features=84, n_units=256, n_classes=10):
    super(RNN, self).__init__()
    self._cnn = CNN(n_features)
    self._fh = nn.Linear(n_features, n_units)
    self._hh = nn.Linear(n_units, n_units)
    self._classifier = nn.Linear(n_units, n_classes)
    self._n_units = n_units

  def forward(self, data):
    N, T, _ = data.size()
    h = Variable(th.zeros(N, self._n_units)).cuda()
    chunks = th.chunk(data, T, 1)
    chunks = map(th.squeeze, chunks)
    chunks = tuple(ch.contiguous().view(N, 1, 28, 28) for ch in chunks)
    pre = []
    for ch in chunks:
      f = self._cnn(ch)
      h = F.tanh(self._fh(f) + self._hh(h))
      c = self._classifier(h)
      pre.append(c)
    pre = tuple(th.unsqueeze(p, 1) for p in pre)
    pre = th.cat(pre, 1)
    return pre
