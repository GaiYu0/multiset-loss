import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self._linear0 = nn.Linear(784, 1024)
    self._linear1 = nn.Linear(1024, 1024)
    self._linear2 = nn.Linear(1024, 1024)
    self._classifier = nn.Linear(1024, 10)

  def forward(self, data):
    size = data.data.size()
    results = []
    for d in torch.chunk(data, size[1], 1):
      d = d.contiguous()
      d = d.view(size[0], -1)
      d = F.relu(self._linear0(d))
      d = F.relu(self._linear1(d))
      d = F.relu(self._linear2(d))
      d = self._classifier(d)
      results.append(d)
    return results
