from collections import Counter
import code
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from absl import app
from absl import flags
from absl import logging

from schema import *

from torch_data import MyDataset
from torchinfo  import summary


FLAGS = flags.FLAGS


# Define model
class MySimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_dim = 64
    self.emb = torch.nn.Embedding(num_embeddings=TRANSFORMER_VOCABULARY,
                                  embedding_dim=self.embedding_dim)
    self.flatten = nn.Flatten()
    self.ln1 = nn.LayerNorm(TRANSFORMER_LENGTH * self.embedding_dim)

    self.dense = nn.LazyLinear(256)
    self.ln2 = nn.LayerNorm(256)
    self.relu = nn.ReLU()
    self.logits = nn.LazyLinear(LABEL_VOCABULARY)

  def forward(self, x):
    x = self.emb(x)
    x = self.flatten(x)
    x = self.ln1(x)

    x = self.dense(x)
    x = self.ln2(x)

    x = self.relu(x)
    x = self.logits(x)
    return x


class MyBiasModel(nn.Module):
  def __init__(self):
    super().__init__()
    #self.bias = nn.Parameter(torch.zeros(LABEL_VOCABULARY))
    self.bias = nn.Parameter(torch.rand(LABEL_VOCABULARY))

  def forward(self, x):
    batch_size = x.size(0)
    return self.bias.unsqueeze(0).repeat(batch_size, 1)



class FixedOneHotModel(nn.Module):
  def __init__(self, output_size, fixed_class_index):
    super(FixedOneHotModel, self).__init__()
    self.output_size = output_size
    self.fixed_class_index = fixed_class_index

    # Create a one-hot tensor
    self.fixed_output = torch.zeros(output_size)
    self.fixed_output[fixed_class_index] = 1

  def forward(self, x):
    batch_size = x.size(0)
    return self.fixed_output.repeat(batch_size, 1)


def main(argv):

  device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
  )
  device = 'cpu'

  print('device: ', device)

  model = MySimpleModel()
  #model = MyBiasModel()
  #model = FixedOneHotModel(LABEL_VOCABULARY, 1807)
  model = model.to(device)
  print(model)
  print()
  summary(model, (1, TRANSFORMER_LENGTH,), dtypes=(torch.int32,))

  ds = MyDataset()
  dl = torch.utils.data.DataLoader(ds, batch_size=64)

  loss_fn = nn.CrossEntropyLoss()
  try:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
  except ValueError:
    # No parameters for the fixed prediction model.
    optimizer = None

  correct, correct_tot = 0, 0
  y_stats = Counter()
  reports = 0
  for batch, (x, y) in enumerate(dl):
    y_stats.update(y.numpy().tolist())
    x = x['board']
    x, y = x.to(device), y.to(device)

    pred = model(x)
    loss = loss_fn(pred, y)
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct_tot += len(x)

    if optimizer:
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    if batch % 10_00 == 0:
      loss, current = loss.item(), (batch + 1) * len(x)
      print(f"{reports}. loss: {loss:>7f}  [{current:>5d}] {100.0 * correct / correct_tot:>6.3f}")
      reports += 1
      #print('c: ', y_stats.most_common(1), y_stats.total())
      correct, correct_tot = 0, 0
      if batch == 100_000:
        break
        with torch.no_grad():
          print()
          for foo in y_stats.most_common(10):
            print('foo: ', foo)
          print()
          for k in sorted(y_stats.keys()):
            print(k, y_stats[k])
          print()
          torch.set_printoptions(threshold=torch.inf)
          bias_values = model.bias

          print('pred: ', pred)
          print('pred shape: ', pred.shape)
          print('bias shape: ', bias_values.shape)

          probabilities = torch.softmax(bias_values, dim=0)
          print('probs: ', probabilities.shape)
          print("Bias values:", bias_values)
          print("Predicted probabilities:", probabilities)
          print('am: ', probabilities.argmax(-1))
          sys.exit(0)



if __name__ == '__main__':
  app.run(main)
