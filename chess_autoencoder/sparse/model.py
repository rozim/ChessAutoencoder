import functools
import time
import warnings

import flax
import jax
import jax.numpy as jnp
import jax.random
import optax
import tensorflow as tf
from absl import app, flags, logging
from clu import metrics
from flax import jax_utils
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.training import common_utils, train_state
from jax import grad, jit, lax, vmap
from ml_collections import config_dict

warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

from schema import (TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)


def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()
  return config

# class Encoder(nn.Module):
#   hidden_size: int = 3
#   embedding_size: int = 7
#   num_classes: int = TRANSFORMER_VOCABULARY

#   def setup(self):
#     self.embedding = nn.Embed(num_embeddings=self.num_classes, features=self.embedding_size)
#     self.dense = nn.Dense(self.hidden_size)

#   def __call__(self, x):
#     x = self.embedding(x)
#     x = jnp.reshape(x, (x.shape[0], -1))  # Flattening using JAX's reshape
#     x = self.dense(x)
#     return x

class Encoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    if x.dtype == jnp.int64:
      x = jnp.asarray(x, dtype=jnp.int32)

    pos_emb = self.param(
      'pos_emb',
      initializers.zeros_init(),
      (1, TRANSFORMER_LENGTH, self.embed_width),
      jnp.float32,
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

  output = enc.apply(params, batch)
  print('output:', output.shape)
  print('output:', output)

  print('read')
  ds = tf.data.TFRecordDataset(['data/mega-2400-00000-of-00100'], 'ZLIB')
  ds = ds.batch(4)
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES))
  ds = ds.as_numpy_iterator()
  for batch in iter(ds):
    #print('board', batch['board'], type(batch['board']), batch['board'].dtype,
    #'bs: ', batch['board'].shape)

    #print('label', batch['label'])
    output = enc.apply(params, batch['board'])
    print('output:', output.shape)
    print('output:', output)
    break






if __name__ == "__main__":
  app.run(main)
