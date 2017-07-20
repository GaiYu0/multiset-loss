from __future__ import division
from argparse import ArgumentParser
import cPickle as pickle
import gzip
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from network import CNN

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=10)
args = parser.parse_args()

th.cuda.set_device(args.gpu)

(training_data, training_labels), (validation_data, validation_labels), (_, _) = \
  pickle.load(gzip.open('mnist.pkl.gz', 'rb'))

training_data, training_labels = th.from_numpy(training_data), th.from_numpy(training_labels)
training_data = training_data.view(-1, 1, 28, 28)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size)

validation_data, validation_labels = th.from_numpy(validation_data), th.from_numpy(validation_labels)
validation_data = validation_data.view(-1, 1, 28, 28)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

model = CNN()
model.cuda()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)

for epoch in range(args.n_epochs):
  for iteration, batch in enumerate(training_loader):
    data, labels = batch
    data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    data = model(data)
    loss = F.nll_loss(F.log_softmax(data), labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (iteration + 1) % args.interval == 0:
      _, data = th.max(data, 1)
      data = th.squeeze(data)
      n_errors = th.sum(data != labels).data[0]
      print 'batch %d training loss %f %d errors encountered' % (iteration + 1, loss.data[0], n_errors)

  n_samples, n_errors = 0, 0
  for batch in validation_loader:
    data, labels = batch
    data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    _, p = th.max(model(data), 1)
    n_samples += data.size()[0]
    n_errors += th.sum(p != labels).data[0]
  error_rate = n_errors / n_samples
  
  print 'epoch %d validation error rate %f' % (epoch, error_rate)

th.save(model.state_dict(), open('pretrained-cnn', 'w'))
# import pdb; pdb.set_trace()
