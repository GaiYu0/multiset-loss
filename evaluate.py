from argparse import ArgumentParser
import joblib
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from network import Network

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--loss', type=str, default='loss')
parser.add_argument('--n_epochs', type=int)
args = parser.parse_args()

def onehot(labels, D):
  size = labels.size()
  results = []
  for l in torch.chunk(labels, size[1], 1):
    l = l.long()
    N = l.size()[0]
    result = torch.zeros(N, D)
    result.scatter_(1, l, 1)
    result = result.view(N, 1, D)
    results.append(result)
  labels = torch.cat(results, 1)
  return labels

training_data, training_labels = joblib.load('training.data')
training_data = torch.from_numpy(training_data)
training_labels = onehot(torch.from_numpy(training_labels), 10)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size)

validation_data, validation_labels = joblib.load('validation.data')
validation_data = torch.from_numpy(validation_data)
validation_labels = onehot(torch.from_numpy(validation_labels), 10)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

test_data, test_labels = joblib.load('test.data')
test_data = torch.from_numpy(test_data)
test_labels = onehot(torch.from_numpy(test_labels), 10)
test_set = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_set, args.batch_size)

model = Network()
model.cuda(args.gpu)
loss_function = getattr(__import__('loss'), args.loss)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(args.n_epochs):
  for index, batch in enumerate(training_loader):
    data, labels = batch
    data, labels = Variable(data.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
    data = model(data)
    loss = loss_function(data, labels)
    loss.backward()
    optimizer.step()

    if (index + 1) % args.interval == 0:
      print 'batch %d training loss %f' % (index + 1, loss.data[0])

  for index, batch in enumerate(validation_loader):
    data, labels = batch
    data, labels = Variable(data.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
    data = model(data)
    loss = loss_function(data, labels)

for index, batch in enumerate(validation_loader):
  data, labels = batch
  data, labels = Variable(data.cuda(args.gpu)), Variable(labels.cuda(args.gpu))
  data = model(data)
  loss = loss_function(data, labels)
  loss.backward()
