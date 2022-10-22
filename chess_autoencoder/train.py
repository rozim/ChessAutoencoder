from contextlib import redirect_stdout
import functools
import json
import logging
import os
import sys
import warnings

import chess

import numpy as np

from absl import app
from absl import flags
from absl import logging as alogging

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Reshape, Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.metrics import Mean, Accuracy, BinaryAccuracy

from encode import encode_board, SHAPE, FLAT_SHAPE
from model import create_models, create_conv_models

FLAGS = flags.FLAGS
flags.DEFINE_integer('dim', 32, '')
flags.DEFINE_integer('batch', 16, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('steps', 100, '')
flags.DEFINE_integer('log_freq', 10, '')
flags.DEFINE_integer('repeat', 0, '')
flags.DEFINE_integer('shuffle', 1024, '')

flags.DEFINE_integer('num_layers', 2, '')
flags.DEFINE_integer('num_filters', 12, '')

flags.DEFINE_string('suffix', '', '')

AUTOTUNE = tf.data.AUTOTUNE


def generate_training_data(fn):
  print(f'Open {fn}')
  with open(fn, 'r') as f:
    for line in f.readlines():
      #fen, _ = line.split(',')
      fen = line.strip()
      board = chess.Board(fen)
      yield encode_board(board)


def generate_training_data2(fn):
  print(f'Open {fn}')
  with open(fn, 'r') as f:
    for line in f.readlines():
      #fen, _ = line.split(',')
      fen = line.strip()
      board = chess.Board(fen)
      yield encode_board(board), fen


def create_dataset(fn):
  gen = functools.partial(generate_training_data, fn)
  ds = tf.data.Dataset.from_generator(gen,
                                      output_signature=(
                                        tf.TensorSpec(shape=SHAPE, dtype=tf.float32)))
  if FLAGS.shuffle:
    ds = ds.shuffle(FLAGS.shuffle)
  ds = ds.batch(FLAGS.batch, drop_remainder=True)
  ds = ds.prefetch(AUTOTUNE)
  return ds

@tf.function
def train_step(x, autoencoder, loss_fn, optimizer):
  fx = tf.reshape(x, (FLAGS.batch, FLAT_SHAPE))
  with tf.GradientTape() as tape:
    y = autoencoder(x, training=True)
    loss_value = loss_fn(y, fx)

  trainable_vars = autoencoder.trainable_variables
  grads = tape.gradient(loss_value, trainable_vars)
  optimizer.apply_gradients(zip(grads, trainable_vars))
  return loss_value, fx, y


def train(autoencoder, ds):
  optimizer = Adam(learning_rate=FLAGS.lr)
  loss_fn = BinaryCrossentropy()

  loss_tracker = Mean(name='loss')
  acc_tracker = Accuracy(name='acc')
  bin_acc_tracker = BinaryAccuracy(name='bin_acc')

  for step, x in enumerate(ds):
    loss_value, fx, y = train_step(x, autoencoder, loss_fn, optimizer)

    loss_tracker.update_state(values=loss_value)
    acc_tracker.update_state(y_true=fx, y_pred=y)
    bin_acc_tracker.update_state(y_true=fx, y_pred=y)

    if step % FLAGS.log_freq == 0:
      print(f'{step:6d} {loss_tracker.result().numpy():.4f} {acc_tracker.result().numpy():.4f} {bin_acc_tracker.result().numpy():.4f}')
      loss_tracker.reset_state()
      acc_tracker.reset_state()
      bin_acc_tracker.reset_state()


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  alogging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  autoencoder, encoder = create_conv_models(FLAGS.dim,
                                            num_filters=FLAGS.num_filters,
                                            num_layers=FLAGS.num_layers)

  if FLAGS.suffix:
    suffix = f'-{FLAGS.suffix}'

  fn = os.path.join(f'model-summary{suffix}.txt')
  with open(fn, 'w') as f:
    with redirect_stdout(f):
      autoencoder.summary()
      print('#')
      encoder.summary()

  fn = argv[1:][0]

  ds = create_dataset(fn)

  if FLAGS.steps:
    ds = ds.take(FLAGS.steps)
  if FLAGS.repeat > 1:
    ds = ds.repeat(FLAGS.repeat)

  train(autoencoder, ds)


  autoencoder.save(f'autoencoder{suffix}.model')
  encoder.save(f'encoder{suffix}.model')


if __name__ == "__main__":
  app.run(main)
