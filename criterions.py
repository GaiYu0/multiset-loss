from pdb import set_trace as st

import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from utilities import onehot, jsd

class Criterion(object):
  def __init__(self):
    super(Criterion, self).__init__()
    self._cuda = False
    self._contextualize = lambda t: t.cuda() if self._cuda else t

  def __call__(self):
    raise NotImplementedError()

  def cuda(self):
    self._cuda = True

class semi_cross_entropy(Criterion):
  def __init__(self):
    super(semi_cross_entropy, self).__init__()

  def __call__(self, data, labels):
    """ A wrapper for the loss function implemented by Jialin and Helen.
    data: (N, T, C)
    labels: (N, T, C) no duplication
    """

    N, T, _ = data.size()
    out = {}
    out['cs'] = th.chunk(data, T, 1)
    out['cs'] = map(th.squeeze, out['cs'])
    out['cs'] = map(F.log_softmax, out['cs'])
    out['cs'] = tuple(th.unsqueeze(ch, 2) for ch in out['cs'])
    out['cs'] = th.cat(out['cs'], 2)
    out['ss'] = th.zeros(N, 1, T)
    out['ss'][:, :, -1] = 1
    out['ss'] = Variable(self._contextualize(out['ss']))
    _, y = th.max(labels, 2)
    y = th.squeeze(y, 2)
    return self._compute_loss(out, y, self._cuda)

  @staticmethod
  def _compute_loss(out, y, use_cuda, discourage=False, backward_kl=False):
    """ By Jialin and Helen. """
    loss = 0.0
    loss_c = 0.0
    loss_s = 0.0
    cs, ss = out['cs'], out['ss']
    B, C, T = cs.size()
    preds = th.max(cs, 1)[1].view(-1, T).detach()
    mask = th.cat([th.zeros(B, C+1).scatter_(1, y.data[:, t].cpu().unsqueeze(1), 1).unsqueeze(2) for t in range(T)],
                  2).sum(2).squeeze()[:, :C]
    for t in range(T):
      ymask = (y.data.cpu()[:, t] != 10).float()
      if use_cuda:
        mask = mask.cuda()
        ymask = ymask.cuda()
      mask = Variable(mask)
      ymask = Variable(ymask)
      ones = th.ones(B, 1)
      if use_cuda:
        ones = ones.cuda()
      ones = Variable(ones)
      if backward_kl:
        log_labels = th.log((mask + 1e-7) / th.max(th.cat((mask.sum(1), ones), 1), 1)[0].expand_as(mask))
        loss_c += ymask * (th.exp(cs[:, :, t]) * (cs[:, :, t] - log_labels)).sum(1)
      elif discourage:
        loss_c = loss_c - ymask * (cs[:, :, t] * mask).sum(1)
      else:
        loss_c = loss_c - ymask * (cs[:, :, t] * mask).sum(1) / th.max(th.cat((mask.sum(1), ones), 1), 1)[0]
      loss_s = loss_s - (ymask * th.log(1 - ss[:,:,t] + 1e-7) + (1 - ymask) * th.log(ss[:,:,t] + 1e-7))
      new_mask = th.zeros(B, C+1).scatter_(1, preds.data.cpu()[:, t].unsqueeze(1), -1)[:, :C] + mask.data.cpu()
      new_mask[new_mask < 0] = 0
      if discourage:
        correct_cls = (new_mask - mask.data.cpu()) < 0
        mask = new_mask - correct_cls.float() * (new_mask == 0).float()
      else:
        mask = new_mask
    loss += (loss_c + loss_s).mean()
    return loss

class alternative_semi_cross_entropy(Criterion):
  def __init__(self):
    super(alternative_semi_cross_entropy, self).__init__()

  def __call__(self, data, labels):
    """ A re-implementation of the loss function implemented by Jialin and Helen.

    data (N, T, C)
    labels (N, T, C) no duplication
    """

    N, T, C = data.size()

    # partition an array with shape (N, T, C) into T arrays with shape (N, C)
    chunks = th.chunk(data, T, 1)
    chunks = map(th.squeeze, chunks)

    chunks = map(F.log_softmax, chunks)

    # one-hot encoding of c_t
    mask = th.sum(labels.data, 1)
    mask = th.squeeze(mask)

    loss = 0
    for index, chunk in enumerate(chunks):
      # 1 / |c_t| \sum_{c \in c_t} \log(p_c)
      loss += th.sum(Variable(mask) * chunk, 1) / (T - index)

      # remove prediction from c_t
      _, p = th.max(chunk, 1)
      onehot_p = onehot(p.data, 10, self._cuda)
      mask = mask - onehot_p

      # in case the prediction does not belong to c_t
      mask = th.max(self._contextualize(th.zeros(mask.size())), mask)

    # likelihodd maximization is equivalent to negtive likelihood minimization
    loss = -th.mean(loss)

    return loss

class jsd_loss(Criterion):
  def __init__(self):
    super(kl_loss, self).__init__()

  def __call__(self, data, labels):
    """
    Instead of computing loss step by step, this loss function aggregates distributions
    along temporal axis and only considers aggregated distributions.
    This loss function consists of 
      - Jensen-Shannon divergence between (aggregated) predicted and targeted distribution
      - An entropy-based regularizer ensuring one-peak behavior of predicted distribution

    data (N, T, C)
    labels (N, T, C) no duplication
    """

    _, T, _ = data.size()
    data = th.chunk(data, T, 1)
    data = map(th.squeeze, data)
    data = map(F.softmax, data)
    data = map(lambda t: th.unsqueeze(t, 1), data)
    data = th.cat(data, 1)

    # aggregates distributions along temporal axis
    data = th.mean(data, 1)
    data = th.squeeze(data)
    labels = th.mean(labels, 1)
    labels = th.squeeze(labels)

    # jsd
    div = jsd(data, labels)

    # for numerical stability
    threshold = th.ones(data.size()) * 1e-5
    threshold = self._contextualize(threshold)
    threshold = Variable(threshold)
    data = th.max(threshold, data)

    # entropy regularizer
    entropy = th.mean(data * th.log(data))

    return div + entropy

class rl_loss(Criterion):
  def __init__(self):
    super(rl_loss, self).__init__()

  def __call__(self, data, labels):
    """ Loss function based on reinforcement learning.

    data (N, T, C)
    labels (N, T, C) no duplication
    """
    
    N, T, C = data.size()

    # partition an array with shape (N, T, C) into T arrays with shape (N, C)
    chunks = th.chunk(data, T, 1)
    chunks = map(th.squeeze, chunks)

    chunks = map(F.log_softmax, chunks)

    # c stands for c_t
    c = th.sum(labels.data, 1)
    c = th.squeeze(c)

    loss = 0
    for index, chunk in enumerate(chunks):
      # compute reward (reward set to 1 if prediction belongs to c_t, -1 otherwise)
      _, p = th.max(chunk, 1)
      onehot_p = onehot(p.data, 10, self._cuda)
      belonging_to = th.sum(c * onehot_p, 1) # whether prediction belongs to c_t
      not_belonging_to = 1 - belonging_to
      not_belonging_to = not_belonging_to.expand_as(onehot_p) # broadcast
      offset = -2 * onehot_p * not_belonging_to
      reward = onehot_p + offset # set reward as -1 for misprediction
      reward = self._contextualize(reward)
      reward = Variable(reward)

      loss += th.sum(chunk * reward)

      c = c - onehot_p # remove prediction from c_t
      threshold = th.zeros(c.size())
      threshold = self._contextualize(threshold)
      c = th.max(threshold, c) # in case of misprediction

    # reward maximization is equivalent to negtive reward minimization
    loss = -loss

    return loss

class ce_loss(Criterion):
  def __init__(self):
    super(ce_loss, self).__init__()

  def __call__(self, data, labels):
    """
    data (N, 1, C)
    labels (N, 1, C) onehot-encoding
    """
    data = th.squeeze(data)
    _, labels = th.max(labels, 2)
    labels = th.squeeze(labels)
    loss = F.cross_entropy(data, labels)
    return loss
