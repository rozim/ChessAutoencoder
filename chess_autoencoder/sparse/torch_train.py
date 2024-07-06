from collections import Counter
import code
import sys
import time

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from absl import app
from absl import flags
from absl import logging

from schema import *
from torch_model import MySimpleModel

from torch_data import MyDataset, FN_TRAIN, FN_TEST
from torchinfo  import summary


from ml_collections.config_dict import ConfigDict
from ml_collections import config_flags



FLAGS = flags.FLAGS

CONFIG = config_flags.DEFINE_config_file('config', 'torch_config.py')

# LOGDIR = flags.DEFINE_string('logdir', '/tmp/logdir', '')
# MODE = flags.DEFINE_enum('mode', 'train', ['train', 'demo'], '')
# START = int(time.time())

def create_optimizer(config: ConfigDict, model: Any) -> Any:
  opt_ctr = {'SGD': torch.optim.SGD}[config.opt_type]
  return opt_ctr(model.parameters(), **config.opt)

def create_model(config: ConfigDict) -> Any:
  model_ctr = {'Simple': MySimpleModel}[config.model_type]
  return model_ctr(**config.model)


def create_train_data(config: ConfigDict) -> torch.utils.data.DataLoader:
  ds_train = MyDataset(FN_TRAIN)
  return torch.utils.data.DataLoader(ds_train, batch_size=config.batch_size)

def create_test_data(config: ConfigDict) -> torch.utils.data.DataLoader:
  ds_test = MyDataset(FN_TEST)
  return torch.utils.data.DataLoader(ds_test, batch_size=config.batch_size)

def create_loss(config: ConfigDict) -> Any:
  return nn.CrossEntropyLoss()


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

  config = CONFIG.value
  print(config)

  model = create_model(config).to(device)
  print(model)
  print()
  summary(model, (1, TRANSFORMER_LENGTH,), dtypes=(torch.int32,))

  dl_train = create_train_data(config)
  dl_test = create_test_data(config)

  loss_fn = create_loss(config)

  optimizer = create_optimizer(config, model)
  print(optimizer)

  correct, correct_tot = 0, 0
  reports = 0
  model.train()
  t1 = time.time()
  train_iter = iter(dl_train)
  for batch in range(config.train_epochs):
    (x, y) = next(train_iter)
    x = x['board']
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = loss_fn(pred, y)
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct_tot += len(x)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % config.train_steps == 0:
      loss, current = loss.item(), (batch + 1) * len(x)
      print(f"{reports}. loss: {loss:>7f}  [{current:>5d}] {100.0 * correct / correct_tot:>6.3f}")
      correct, correct_tot = 0, 0
      reports += 1



if __name__ == '__main__':
  app.run(main)
