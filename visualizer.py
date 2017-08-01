import numpy as np

class Visualizer(object):
  def __init__(self, visdom, options={}, start=0):
    self._visdom = visdom
    self._window = None
    self._options = options
    self._start = start

  def extend(self, s, clear=True):
    X = np.arange(self._start, self._start + len(s))
    Y = np.array(s)
    if self._window:
      self._window = self._visdom.line(Y, X, self._window, opts=self._options, update='append')
    else:
      self._window = self._visdom.line(Y, X, opts=self._options)
    self._start += len(s)
    if clear:
      del s[:]
