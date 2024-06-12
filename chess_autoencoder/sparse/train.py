
from ml_collections import config_dict

from absl import app
from absl import flags
from absl import logging

import flax
from flax import linen as nn
from flax.training import common_utils
from flax import jax_utils
from flax.training import train_state
from flax import struct

import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, lax
import jax.random

from clu import metrics

import optax

import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)




@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics


def main(argv):
  pass

if __name__ == "__main__":
  app.run(main)
