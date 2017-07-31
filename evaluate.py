from argparse import ArgumentParser
import joblib
import torch as th
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from loss import loss as loss_function
from network import CNN, RNN
from utilities import onehot, onehot_sequence, n_matches

parser = ArgumentParser()
parser.add_argument('--cnn-path', type=str, default='pretrained-cnn')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--n-epochs', type=int, default=10)
parser.add_argument('--n-units', type=int, default=0) # deactivate RNN if n-units is 0
parser.add_argument('--pretrained-cnn', action='store_true', default=False)
args = parser.parse_args()
print args

if args.use_gpu:
    th.cuda.set_device(args.gpu)

training_data, training_labels = joblib.load('training-%d.data' % args.n)
training_data = th.from_numpy(training_data)
training_labels = onehot_sequence(th.from_numpy(training_labels), 10, args.use_gpu)
training_set = TensorDataset(training_data, training_labels)
training_loader = DataLoader(training_set, args.batch_size)

validation_data, validation_labels = joblib.load('validation-%d.data' % args.n)
validation_data = th.from_numpy(validation_data)
validation_labels = onehot_sequence(th.from_numpy(validation_labels), 10, args.use_gpu)
validation_set = TensorDataset(validation_data, validation_labels)
validation_loader = DataLoader(validation_set, args.batch_size)

test_data, test_labels = joblib.load('test-%d.data' % args.n)
test_data = th.from_numpy(test_data)
test_labels = onehot_sequence(th.from_numpy(test_labels), 10, args.use_gpu)
test_set = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_set, args.batch_size)

cnn_path = args.cnn_path if args.pretrained_cnn else None
model = RNN(args.n_units, 10, cnn_path)
if args.use_gpu:
    model.cuda()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(args.n_epochs):
    for index, batch in enumerate(training_loader):
        data, labels = batch
        if args.use_gpu:
            data, labels = Variable(data.cuda()), Variable(labels.cuda())
        else:
            data, labels = Variable(data), Variable(labels)
        data = model(data)
        loss = loss_function(data, labels, args.use_gpu)
        optimizer.zero_grad()
        loss.backward()
        if args.pretrained_cnn:
            model.cnn_zero_grad()
        optimizer.step()

        if (index + 1) % args.interval == 0:
            ns = float(data.size()[0])
            nm = float(n_matches(data, labels))
            ratio = nm / ns
            print 'batch %d training loss %f ratio of matching %f' % (index + 1, loss.data[0], ratio)

    ns, nm = 0.0, 0.0
    for index, batch in enumerate(validation_loader):
        data, labels = batch
        if args.use_gpu:
            data, labels = Variable(data.cuda()), Variable(labels.cuda())
        data = model(data)
        ns += data.size()[0]
        nm += n_matches(data, labels)
        ratio = nm / ns
        print 'epoch %d ratio of matching %f' % (epoch + 1, ratio)

ns, nm = 0.0, 0.0
for index, batch in enumerate(test_loader):
    data, labels = batch
    if args.use_gpu:
        data, labels = Variable(data.cuda()), Variable(labels.cuda())
    data = model(data)
    ns += data.size()[0]
    nm += n_matches(data, labels)
    print 'epoch %d total number of errors %d' % (epoch + 1, ratio)
# import pdb; pdb.set_trace()
