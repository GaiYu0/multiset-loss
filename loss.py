import torch
import torch.nn.functional as F

def jsd(p, q):
  m = (p + q) / 2
  d = F.kl_div(p, m) + F.kl_div(m, q)
  return d

def loss(data, labels):
  data = sum(F.softmax(d) for d in data) / len(data)
  size = labels.data.size()
  labels = torch.chunk(labels, size[1], 1)
  labels = sum(labels) / len(data)
  loss = F.kl_div(data, labels)
  return loss
