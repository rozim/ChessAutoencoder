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
from model import AutoEncoder, Encoder
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)


class TestTrain:

  def test_nothing(self):
    pass
