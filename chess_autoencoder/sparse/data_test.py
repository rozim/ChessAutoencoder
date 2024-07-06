import time
from typing import Any, Callable, Dict
import functools

import flax
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tensorflow as tf
from chex import assert_rank, assert_shape, assert_type
from flax.training import common_utils, train_state

import train
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_VOCABULARY)

def test_shape_and_range_of_data():
  bs = 1024
  files = 'data/mega-2400-00000-of-00100'
  ds = tf.data.TFRecordDataset(files, 'ZLIB')
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES))
  ds = ds.batch(bs)
  ds = ds.take(1)
  ds = ds.as_numpy_iterator()
  batch = next(iter(ds))
  board = batch['board']
  label = batch['label']
  assert_shape(board, (bs, 69))
  assert_shape(label, (bs, 1))
  assert jnp.all(board >= 0)
  assert jnp.all(board < TRANSFORMER_VOCABULARY)
  assert jnp.all(label >= 0)
  assert jnp.all(label < LABEL_VOCABULARY)
