import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import gzip
import json


from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

FN_TRAIN = '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-train.jsonl.gz'
FN_TEST = '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-test.jsonl.gz'

class MyDataset(torch.utils.data.IterableDataset):
  def __init__(self, fn: str):
    self.f = gzip.GzipFile(fn, 'rb')

  def __iter__(self):
    for line in self.f:
      j_obj = json.loads(line)
      yield ({'board': torch.tensor(j_obj['board'])},
             torch.tensor(j_obj['label']))



def main(argv):

  ds = MyDataset(FN_TRAIN)

  dl = torch.utils.data.DataLoader(ds)

  for ent in dl:
    print(ent)
    break

  ds = MyDataset(FN_TEST)
  dl = torch.utils.data.DataLoader(ds)

  for ent in dl:
    print(ent)
    break


if __name__ == '__main__':
  app.run(main)
