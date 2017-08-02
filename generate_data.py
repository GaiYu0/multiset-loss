# wget http://deeplearning.net/data/mnist/mnist.pkl.gz
from pdb import set_trace as st

from argparse import ArgumentParser
import cPickle as pickle
import gzip
import joblib
import numpy as np

import os
datapath = os.environ.get('datapath', '')

parser = ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--path', type=str, default='seq.data')
parser.add_argument('--replace', action='store_true', default=False)
parser.add_argument('--size', type=int)
parser.add_argument('--source', type=str, default='training')
args = parser.parse_args()

source_index = {'training': 0, 'validation': 1, 'test': 2}[args.source]
data, labels = pickle.load(gzip.open(datapath + 'mnist.pkl.gz', 'rb'))[source_index]
data = (data - data.mean()) / data.std()

if args.replace:
  categories = np.random.choice(np.arange(10), (args.size, args.n), replace=True)
else:
  assert args.n < 11
  rows = tuple(np.random.choice(np.arange(10), (1, args.n), replace=False) \
               for _ in range(args.size))
  categories = np.vstack(rows)

indices = np.copy(categories)
candidate_sets = tuple(np.argwhere(labels == i).flatten() for i in range(10))
for i, candidate_set in enumerate(candidate_sets):
  mask = indices == i
  n = np.sum(mask)
  indices[mask] = np.random.choice(candidate_set, n, replace=True)
data = data[indices]
labels = categories

joblib.dump((data, labels), open(args.path, 'wb'))
