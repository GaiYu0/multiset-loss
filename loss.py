import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from utilities import onehot

'''
def loss(data, labels):
  # data (N, T, C)
  # labels (N, T, C) no duplication
  data = F.log_softmax(data)
  entropy = th.mean(th.exp(data) * data)
  data = th.sum(data, 1)
  data = th.squeeze(data)
  masks = th.sum(labels, 1)
  masks = th.squeeze(masks)
  likelihood = -th.mean(masks * data)
  value = entropy + likelihood
  return value
'''

'''
def loss(data, labels):
# import pdb; pdb.set_trace()
  # data (N, T, C)
  # labels (N, T, C) no duplication

  N, T, C = data.size()
  chunks = th.chunk(data, T, 1)
  chunks = map(th.squeeze, chunks)
  chunks = map(F.log_softmax, chunks)
# print map(entropy, tuple(ch.data.cpu() for ch in chunks))
  mask = th.sum(labels.data, 1)
  mask = th.squeeze(mask)
  loss = 0

  for index, chunk in enumerate(chunks):
    loss += th.sum(Variable(mask) * chunk, 1) / (T - index)
    _, p = th.max(chunk, 1)
    onehot_p = onehot(p.data, 10)
    mask = mask - onehot_p
    mask = th.max(th.zeros(mask.size()).cuda(), mask)

  loss = -th.mean(loss)
  return loss
'''

def loss(data, labels):
# import pdb; pdb.set_trace()
  # data (N, T, C)
  # labels (N, T, C) no duplication
  N, T, _ = data.size()
  out = {}
  out['cs'] = th.chunk(data, T, 1)
  out['cs'] = map(th.squeeze, out['cs'])
  out['cs'] = map(F.log_softmax, out['cs'])
  out['cs'] = tuple(th.unsqueeze(ch, 2) for ch in out['cs'])
  out['cs'] = th.cat(out['cs'], 2)
  out['ss'] = th.zeros(N, 1, T)
  out['ss'][:, :, -1] = 1
  out['ss'] = Variable(out['ss'].cuda())
  out['ss'] = None
  _, y = th.max(labels, 2)
  y = th.squeeze(y)
  return compute_loss(out, y, True)

def compute_loss(out, y, use_cuda, discourage=False, backward_kl=False):
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
#   loss_s = loss_s - (ymask * th.log(1 - ss[:,:,t] + 1e-7) + (1 - ymask) * th.log(ss[:,:,t] + 1e-7))
    new_mask = th.zeros(B, C+1).scatter_(1, preds.data.cpu()[:, t].unsqueeze(1), -1)[:, :C] + mask.data.cpu()
    new_mask[new_mask < 0] = 0
    if discourage:
      correct_cls = (new_mask - mask.data.cpu()) < 0
      mask = new_mask - correct_cls.float() * (new_mask == 0).float()
    else:
      mask = new_mask
  loss += (loss_c + loss_s).mean()
  return loss
