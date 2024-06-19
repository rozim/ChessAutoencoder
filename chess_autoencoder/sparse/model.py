import functools
import time
import warnings
from typing import Any, Callable, Dict

import flax
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
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
from ml_collections import config_dict, config_flags

warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

from chex import assert_rank, assert_shape

from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)


class Encoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    assert_rank(x, 2)
    assert_shape(x, (None, TRANSFORMER_LENGTH))

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
    x = jax.nn.sigmoid(x)

    # shape: (latent_dim)
    return x


class Decoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    assert_rank(x, 2)

    assert_shape(x, (None, self.latent_dim))

    x = nn.Dense(name='decoding', features=(TRANSFORMER_LENGTH * self.embed_width))(x)
    x = x.reshape((x.shape[0], TRANSFORMER_LENGTH, self.embed_width))
    x = nn.Dense(name='logits', features=(LABEL_VOCABULARY))(x)
    assert_shape(x, (None, TRANSFORMER_LENGTH, LABEL_VOCABULARY))
    return x

class AutoEncoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    encoder = Encoder(self.latent_dim, self.embed_width)
    decoder = Decoder(self.latent_dim, self.embed_width)
    z = encoder(x)
    y = decoder(z)
    return y
