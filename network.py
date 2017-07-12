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
    for d in 
      d = F.relu(self._linear0(d))
      d = F.relu(self._linear1(d))
      d = F.relu(self._linear2(d))
      d = self._classifier(d)
