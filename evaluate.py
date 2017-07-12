from argparse import ArgumentParser
from torch.autograd import Variable
from torch.optim import SGD
from network import Network

parser = ArgumentParser()
parser.add_argument('--loss', type=str)
args = parser.parse_args()

model = Network()
loss_function = getattr(__import__('loss'), args.loss)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

for i, batch in enumerate(training_loader):
  data, labels = batch
  data, labels = Variable(data), Variable(labels) 
  data = model(data)
  loss = loss_function(data, labels)
  loss.backward()
  optimizer.step()
