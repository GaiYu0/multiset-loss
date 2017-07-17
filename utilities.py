import torch as th

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

def n_errors(data, labels):
  data, labels = data.data, labels.data
  _, p = th.max(data, 2)
  p = th.squeeze(p)
  onehot_p = onehot_sequence(p, 10)
  n = th.sum(th.abs(onehot_p - labels))
  return n
