from __future__ import division

from argparse import ArgumentParser
import cPickle as pickle
import gzip
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from network import CNN

parser = ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

partitions = ('training', 'validation', 'test')
data = pickle.load(gzip.open('mnist.pkl.gz'))
data = dict(zip(partitions, data))

cnn = CNN()
state_dict = th.load(args.path)
cnn.load_state_dict(state_dict)
cnn.cuda()

batch_size = 1024
data_loaders = {}
for key, value in data.items():
    value = map(th.from_numpy, value)
    dataset = TensorDataset(*value)
    data_loaders[key] = DataLoader(dataset, batch_size)
    
def n_matches(p, labels):
    _, p = th.max(p, 1)
    p = th.squeeze(p)
    indicator = p == labels
    n = th.sum(indicator.double())
    n = n.data[0]
    return n
    
for partition, data_loader in data_loaders.items():
    ns, nm = 0, 0
    for batch in data_loader:
        data, labels = batch
        data = data.view(-1, 1, 28, 28)
        data, labels = Variable(data.cuda()), Variable(labels.cuda())
        data = cnn(data)
        ns += data.size()[0]
        nm += n_matches(data, labels)
    ratio = nm / ns
    print '%s %3f' % (partition, ratio)
