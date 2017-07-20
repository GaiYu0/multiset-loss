# wget http://deeplearning.net/data/mnist/mnist.pkl.gz

from argparse import ArgumentParser
import cPickle as pickle
import gzip
import joblib
import numpy as np

parser = ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--path', type=str)
parser.add_argument('--replace', action='store_true', default=False)
parser.add_argument('--size', type=int)
parser.add_argument('--source', type=str, default='training')
args = parser.parse_args()

source_index = {'training': 0, 'validation': 1, 'test': 2}[args.source]
data, labels = pickle.load(gzip.open('mnist.pkl.gz', 'rb'))[source_index]
data = (data - data.mean()) / data.std()

if args.replace:
  shape = (args.size, args.n)
  indices = np.random.choice(np.arange(10), shape)
else:
  rows = (np.random.choice(np.arange(10), (1, args.n), replace=False) \
    for i in range(args.size))
  indices = np.vstack(rows)
indexed_data = data[indices]
indexed_labels = labels[indices]

joblib.dump((indexed_data, indexed_labels), open(args.path, 'wb'))
