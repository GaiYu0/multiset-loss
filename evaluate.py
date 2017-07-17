from argparse import ArgumentParser
import joblib
import torch as th
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from network import Network
from utilities import onehot, onehot_sequence, n_errors

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--loss', type=str, default='loss')
parser.add_argument('--n_epochs', type=int)
args = parser.parse_args()

th.cuda.set_device(args.gpu)

training_data, training_labels = joblib.load('training.data')
training_data = th.from_numpy(training_data)
# training_labels = th.from_numpy(training_labels)
training_labels = onehot_sequence(th.from_numpy(training_labels), 10)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size)

validation_data, validation_labels = joblib.load('validation.data')
validation_data = th.from_numpy(validation_data)
# validation_labels = th.from_numpy(validation_labels)
validation_labels = onehot_sequence(th.from_numpy(validation_labels), 10)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

test_data, test_labels = joblib.load('test.data')
test_data = th.from_numpy(test_data)
# test_labels = th.from_numpy(test_labels)
test_labels = onehot_sequence(th.from_numpy(test_labels), 10)
test_set = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_set, args.batch_size)

model = Network()
model.cuda()
loss_function = getattr(__import__('loss'), args.loss)
optimizer = Adam(model.parameters(), lr=0e-2)
# optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(args.n_epochs):
  for index, batch in enumerate(training_loader):
    data, labels = batch
    data, labels = Variable(data.cuda()), Variable(labels.cuda())
    data = model(data)
    loss = loss_function(data, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (index + 1) % args.interval == 0:
      print 'batch %d training loss %f' % (index + 1, loss.data[0])

  total_n_errors = 0
  for index, batch in enumerate(validation_loader):
    data, labels = batch
    data, labels = Variable(data.cuda()), Variable(labels.cuda())
    data = model(data)
    total_n_errors += n_errors(data, labels)
  print 'epoch %d total number of errors %d' % (epoch + 1, total_n_errors)

total_n_errors = 0
for index, batch in enumerate(test_loader):
  data, labels = batch
  data, labels = Variable(data.cuda()), Variable(labels.cuda())
  data = model(data)
  total_n_errors += n_errors(data, labels)
print 'epoch %d total number of errors %d' % (epoch + 1, total_n_errors)
