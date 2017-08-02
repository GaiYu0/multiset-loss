from __future__ import division

from pdb import set_trace as st

from argparse import ArgumentParser
import joblib
import torch as th
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import visdom
from network import CNN, RNN
from utilities import onehot, onehot_sequence, n_matches
from visualizer import VisdomVisualizer, TensorboardVisualizer

parser = ArgumentParser()
parser.add_argument('--pretrained-cnn-path', type=str, default='pretrained-cnn')
parser.add_argument('--cnn-path', type=str, default='')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--datapath', type=str, default='data')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--interval', type=int, default=100)
# --criterion=semi_cross_entropy/alternative_semi_cross_entropy/jsd_loss/rl_loss
parser.add_argument('--criterion', type=str, default='semi_cross_entropy')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--n-epochs', type=int, default=10)
parser.add_argument('--n-units', type=int, default=0) # disable RNN if n-units is 0
parser.add_argument('--pretrained-cnn', action='store_true', default=False)
parser.add_argument('--tensorboard-log', type=str, default='')
parser.add_argument('--tensorboard-path', type=str, default='tensorboard-log')
parser.add_argument('--tensorboard-postfix', type=str, default='')
args = parser.parse_args()
print args

if args.gpu > -1:
  cuda = True
  th.cuda.set_device(args.gpu)
else:
  cuda = False

training_data, training_labels = joblib.load('%s/training-%d.data' % (args.datapath, args.n))
training_data = th.from_numpy(training_data)
training_labels = onehot_sequence(th.from_numpy(training_labels), 10, cuda)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size, shuffle=True)

validation_data, validation_labels = joblib.load('%s/validation-%d.data' % (args.datapath, args.n))
validation_data = th.from_numpy(validation_data)
validation_labels = onehot_sequence(th.from_numpy(validation_labels), 10, cuda)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

test_data, test_labels = joblib.load('%s/test-%d.data' % (args.datapath, args.n))
test_data = th.from_numpy(test_data)
test_labels = onehot_sequence(th.from_numpy(test_labels), 10, cuda)
test_set = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_set, args.batch_size)

cnn_path = args.pretrained_cnn_path if args.pretrained_cnn else None
model = RNN(args.n_units, 10, cnn_path, cuda)
if args.gpu > -1:
  model.cuda()
criterion = getattr(__import__('criterions'), args.criterion)()
if args.gpu > -1:
  criterion.cuda()
optimizer = Adam(model.parameters(), lr=1e-3)
vis = visdom.Visdom()
tb_path = args.tensorboard_path
if args.tensorboard_log:
  tb_path += '/%s' % args.tensorboard_log
TensorboardVisualizer.configure(tb_path)
loss_list = []
loss_vis = VisdomVisualizer(vis, {'title': 'loss'})
loss_tb = TensorboardVisualizer('loss' + args.tensorboard_postfix)
ratio_list = []
ratio_vis = VisdomVisualizer(vis, {'title': 'training ratio of matching'})
ratio_tb = TensorboardVisualizer('training-ratio-of-matching' + args.tensorboard_postfix)
validation_vis = VisdomVisualizer(vis, {'title': 'validation ratio of matching'})
validation_tb = TensorboardVisualizer('validation-ratio-of-matching' + args.tensorboard_postfix)

for epoch in range(args.n_epochs):
  for index, batch in enumerate(training_loader):
    data, labels = batch
    if args.gpu > -1:
      data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    data = model(data)
    loss = criterion(data, labels)
    optimizer.zero_grad()
    if args.n_units > 0 or not args.pretrained_cnn:
      loss.backward()
    optimizer.step()

    loss_list.append(loss.data[0])

    ns = data.size()[0]
    nm = n_matches(data, labels)
    ratio = nm / ns
    ratio_list.append(ratio)

    if (index + 1) % args.interval == 0:
      loss_vis.extend(loss_list, True)
      loss_tb.extend(loss_list, True)
      ratio_vis.extend(ratio_list, True)
      ratio_tb.extend(ratio_list, True)
      print 'batch %d training loss %f ratio of matching %f' % (index + 1, loss.data[0], ratio)

  ns, nm = 0.0, 0.0
  for index, batch in enumerate(validation_loader):
    data, labels = batch
    if args.gpu > -1:
      data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    data = model(data)
    ns += data.size()[0]
    nm += n_matches(data, labels)
  ratio = nm / ns
  validation_vis.extend((ratio,))
  validation_tb.extend((ratio,))
  print 'epoch %d ratio of matching %f' % (epoch + 1, ratio)

ns, nm = 0.0, 0.0
for index, batch in enumerate(test_loader):
  data, labels = batch
  if args.gpu > -1:
    data, labels = data.cuda(), labels.cuda()
  data, labels = Variable(data), Variable(labels)
  data = model(data)
  ns += data.size()[0]
  nm += n_matches(data, labels)
ratio = nm / ns
print 'ratio of matching %f' % ratio

if args.model_path:
  th.save(model.state_dict(), args.model_path)
if args.cnn_path:
  th.save(model._cnn.state_dict(), args.cnn_path)
