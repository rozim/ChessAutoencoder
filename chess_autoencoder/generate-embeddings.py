import functools
import sys
import os
import json

import chess

import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Reshape, Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.metrics import Mean, Accuracy, BinaryAccuracy

import sqlitedict


from encode import encode_board, SHAPE, FLAT_SHAPE
from model import create_models

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch', 16, '')


def generate_training_data2(fn):
  print(f'Open {fn}')
  with open(fn, 'r') as f:
    for line in f.readlines():
      #fen, _ = line.split(',')
      fen = line.strip()
      board = chess.Board(fen)
      yield encode_board(board), fen


def main(argv):
  fn = argv[1:][0]
  encoder = tf.keras.models.load_model('encoder.model')

  db = sqlitedict.open(filename='embeddings.sqlite',
                       flag='c',
                       encode=json.dumps,
                       decode=json.loads)
  stack = []
  fens = []
  n = 0
  for encoded, fen in generate_training_data2(fn):
    if n % 5000 == 0:
      print('ex', n)
    n += 1
    stack.append(encoded)
    fens.append(fen)
    if len(stack) == FLAGS.batch:
      batch = np.stack(stack)
      embeddings = encoder(batch, training=False)

      for k, v in zip(fens, tf.unstack(embeddings)):
        db[k] = v.numpy().tolist()
      stack = []
      fens = []


  if len(stack) > 0:
    batch = np.stack(stack)
    embeddings = encoder(batch, training=False)
    for k, v in zip(fens, tf.unstack(embeddings)):
      db[k] = v.numpy().tolist()

  db.commit()
  db.close()
  print('done')


if __name__ == "__main__":
  app.run(main)
