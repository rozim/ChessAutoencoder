from collections import Counter
import code
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from absl import app
from absl import flags
from absl import logging

from schema import *

from torch_data import MyDataset, FN_TRAIN, FN_TEST
from torchinfo  import summary


FLAGS = flags.FLAGS

from ml_collections import config_dict
from ml_collections import config_flags

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

if __name__ == '__main__':
  app.run(main)
