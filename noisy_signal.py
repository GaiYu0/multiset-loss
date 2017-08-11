from __future__ import division
from pdb import set_trace as st
from argparse import ArgumentParser
import cPickle as pickle
import gzip
import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from network import CNN
from utilities import onehot

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--noise', type=float, default=0.75)
args = parser.parse_args()

if args.gpu < 0:
  cuda = False
else:
  cuda = True
  th.cuda.set_device(args.gpu)

(training_data, training_labels), (validation_data, validation_labels), (_, _) = \
  pickle.load(gzip.open('mnist.pkl.gz', 'rb'))

training_data, training_labels = th.from_numpy(training_data), th.from_numpy(training_labels)
training_data = training_data.view(-1, 1, 28, 28)
training_labels = th.unsqueeze(training_labels, 1)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size)

validation_data, validation_labels = th.from_numpy(validation_data), th.from_numpy(validation_labels)
validation_data = validation_data.view(-1, 1, 28, 28)
validation_labels = th.unsqueeze(validation_labels, 1)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

model = CNN()
if cuda:
  model.cuda()
criterion = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(args.n_epochs):
  for iteration, batch in enumerate(training_loader):
    data, labels = batch
    if cuda:
      data, labels = data.cuda(), labels.cuda()

    noisy_labels = th.zeros(labels.size())
    noisy_labels.copy_(labels)
    n_noises = int(args.noise * labels.size()[0])
    noise = th.from_numpy(np.random.choice(np.arange(1, 10), n_noises))
    if cuda:
      noise = noise.cuda()
    if n_noises > 0:
      noisy_labels[:n_noises] = (labels[:n_noises] + noise) % 10
    noisy_labels = onehot(noisy_labels.long(), 10, cuda).float()

    data, noisy_labels = Variable(data), Variable(noisy_labels)
    data = model(data)
    data = F.softmax(data)
    loss = criterion(data, noisy_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (iteration + 1) % args.interval == 0:
      _, data = th.max(data, 1)
      data = th.squeeze(data).data
      n_errors = th.sum(data != labels)
      print 'batch %d training loss %f %d errors encountered' % (iteration + 1, loss.data[0], n_errors)

  n_samples, n_errors = 0, 0
  for batch in validation_loader:
    data, labels = batch
    if cuda:
      data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    _, p = th.max(model(data), 1)
    n_samples += data.size()[0]
    n_errors += th.sum(p != labels).data[0]
  error_rate = n_errors / n_samples
  
  print 'epoch %d validation error rate %f' % (epoch, error_rate)
