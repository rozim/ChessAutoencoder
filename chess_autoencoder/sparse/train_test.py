import time
from typing import Any, Callable, Dict

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
from model import AutoEncoderBoardHead, Encoder
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)


class TestTrain:

  def test_nothing(self):
    pass

  def test_accuracy_non_zero(self):
    predictions = jnp.array([1, 2, 0, 2, 2, 2])
    labels = jnp.array(     [1, 0, 0, 9, 2, 2])
    assert train.accuracy_non_zero(predictions, labels) == 0.75
