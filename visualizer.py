from pdb import set_trace as st
import numpy as np

class Visualizer(object):
  def __init__(self, start):
    super(Visualizer, self).__init__()
    self._start = start

  def extend(self, s, clear=False):
    self._extend(s)
    self._start += len(s)
    if clear:
      del s[:]

class VisdomVisualizer(Visualizer):
  def __init__(self, visdom, options={}, start=0):
    super(VisdomVisualizer, self).__init__(start)
    self._visdom = visdom
    self._window = None
    self._options = options

  def _extend(self, s):
    X = np.arange(self._start, self._start + len(s))
    Y = np.array(s)
    if self._window:
      self._window = self._visdom.line(Y, X, self._window, opts=self._options, update='append')
    else:
      self._window = self._visdom.line(Y, X, opts=self._options)

# TODO use Logger
class TensorboardVisualizer(Visualizer):
  _logger = __import__('tensorboard_logger')
  def __init__(self, name, start=0):
    super(TensorboardVisualizer, self).__init__(start)
    self._name = name

  @staticmethod
  def configure(path):
    TensorboardVisualizer._logger.configure(path)

  def _extend(self, s):
    for index, value in enumerate(s):
      self._logger.log_value(self._name, value, self._start + index)
