import time

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
from flax.linen import initializers

import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, lax
import jax.random

from clu import metrics

import optax

import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

from schema import TRANSFORMER_LENGTH, TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY


def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()
  return config


class Encoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    pos_emb = self.param(
      'pos_emb',
      initializers.zeros_init(),
      (TRANSFORMER_LENGTH, self.embed_width), jnp.float32,
    )

    x = nn.Embed(num_embeddings=TRANSFORMER_VOCABULARY,
                 features=self.embed_width,
                 dtype=jnp.float32)(x)

    x += pos_emb

    x = x.reshape((x.shape[0], -1))  # flatten

    # shape: (TRANSFORMER_LENGTH * embed_with)
    x = nn.Dense(name='encoding', features=self.latent_dim)(x)

    # shape: (latent_dim)
    return x

def main(argv):
  print(get_config())
  #t = TrainState()
  #m = Metrics()
  rng = jax.random.PRNGKey(int(time.time()))
  enc = Encoder()
  sample_x = jnp.ones((1,) + TRANSFORMER_SHAPE, jnp.int32)
  sample_x = jax.random.randint(key=rng,
                                shape=((1,) + TRANSFORMER_SHAPE),
                                minval=0,
                                maxval=TRANSFORMER_VOCABULARY,
                                dtype=jnp.int32)
  params = enc.init(rng, sample_x)
  print('params: ', params)
  print()
  print(enc.tabulate(rng, sample_x))
  print()
  print('enc: ', enc)
  print()
  batch = sample_x
  # batch = jax.random.randint(key=rng,
  #                            shape=((1,) + TRANSFORMER_SHAPE),
  #                            minval=0,
  #                            maxval=TRANSFORMER_VOCABULARY,
  #                            dtype=jnp.int32)

  output = enc.apply(params, batch)
  print('output:', output.shape)
  print('output:', output)




if __name__ == "__main__":
  app.run(main)
