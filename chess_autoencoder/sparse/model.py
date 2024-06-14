import functools
import time
import warnings
from typing import Any, Callable, Dict

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
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY, LABEL_VOCABULARY)

from chex import assert_shape, assert_rank


def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()
  return config


class Encoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    assert_rank(x, 2)
    assert_shape(x, (None, TRANSFORMER_LENGTH))

    print('e/x0', x.shape)
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


class Decoder(nn.Module):
  latent_dim: int = 3
  embed_width: int = 2

  @nn.compact
  def __call__(self, x):
    print('x0', x.shape)
    assert_rank(x, 2)

    assert_shape(x, (None, self.latent_dim))

    x = nn.Dense(name='decoding', features=(TRANSFORMER_LENGTH * self.embed_width))(x)
    print('x1', x.shape)
    x = x.reshape((x.shape[0], TRANSFORMER_LENGTH, self.embed_width))
    print('x2', x.shape)
    x = nn.Dense(name='logits', features=(LABEL_VOCABULARY))(x)
    print('x3', x.shape)
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


def compute_metrics(*, logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'accuracy': accuracy,
  }
  return metrics

@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    label: jnp.ndarray
):
  def _loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
    return loss.mean(), logits

  gradient_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, logits), grads = gradient_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=label)
  metrics['loss'] = loss
  return state, metrics

def main(argv):
  print(get_config())
  latent_dim = 2
  embed_width = 3
  #t = TrainState()
  #m = Metrics()
  rng = jax.random.PRNGKey(int(time.time()))
  rng, rnd_enc, rnd_dec, rnd_ae = jax.random.split(rng, num=4)
  enc = Encoder(latent_dim=latent_dim, embed_width=embed_width)
  dec = Decoder(latent_dim=latent_dim, embed_width=embed_width)
  ae = AutoEncoder(latent_dim=latent_dim, embed_width=embed_width)
  sample_x = jax.random.randint(key=rng,
                                shape=((1,) + TRANSFORMER_SHAPE),
                                minval=0,
                                maxval=TRANSFORMER_VOCABULARY,
                                dtype=jnp.int32)
  sample_z = jax.random.uniform(key=rng,
                                shape=(1, latent_dim),
                                minval=-1.0,
                                maxval=1.0,
                                dtype=jnp.float32)

  enc_variables = enc.init(rnd_enc, sample_x)
  print('enc_variables: ', enc_variables)
  dec_variables = dec.init(rnd_dec, sample_z)
  print('dec_variables: ', dec_variables)
  ae_variables = ae.init(rnd_ae, sample_x)
  print('ae_variables: ', ae_variables)
  print()
  print(enc.tabulate(rnd_enc, sample_x))
  print()
  print(dec.tabulate(rnd_dec, sample_z))
  print()
  print(ae.tabulate(rnd_ae, sample_x))
  print()
  print('enc: ', enc)
  print()
  print('dec: ', dec)
  print()
  print('ae: ', ae)
  print()
  batch = sample_x

  output = enc.apply(enc_variables, batch)
  print('output:', output.shape)
  print('output:', output)

  print('read')
  batch_size = 4
  ds = tf.data.TFRecordDataset(['data/mega-2400-00000-of-00100'], 'ZLIB')
  ds = ds.batch(batch_size)
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES))
  ds = ds.as_numpy_iterator()
  for batch in iter(ds):
    #print('board', batch['board'], type(batch['board']), batch['board'].dtype,
    #'bs: ', batch['board'].shape)

    #print('label', batch['label'])
    output = enc.apply(enc_variables, batch['board'])
    print('output:', output.shape)
    print('output:', output)

    output2 = dec.apply(dec_variables, output)
    print('output2:', output2.shape)
    print('output2:', output2)
    labels = batch['label']
    labels = batch['label'].reshape([batch_size, 1])

    print('labels:', labels.shape)
    print('labels/1:', labels)
    print('labels/2:', labels[..., None])
    print('max', jnp.max(output2, axis=-1, keepdims=True))



    logits = output2
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    print('z1', logits_max)
    logits -= jax.lax.stop_gradient(logits_max)
    print('z2', logits)
    print('z3/0', logits.shape, labels[..., None].shape)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    print('z3/a', label_logits)
    print('z3/b', jnp.take_along_axis(logits, labels[..., None], axis=-1))
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    print('z4', log_normalizers)

    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=output2,
      labels=labels)
    print('loss=', loss)
    print('loss=', loss.mean())

    optimizer = optax.adamw(learning_rate=0.01)
    # ts_dec = train_state.TrainState.create(
    #   apply_fn=dec.apply,
    #   tx=optimizer,
    #   params=dec_variables['params']
    # )
    #print('ts', dec_enc)
    #state, metrics = train_step(state, batch['board'], batch['label'])

    break






if __name__ == "__main__":
  app.run(main)
