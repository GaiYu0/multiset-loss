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
