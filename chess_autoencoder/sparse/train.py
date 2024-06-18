import functools
import os
import time
import warnings

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
from flax.training import common_utils, train_state
from jax import grad, jit, lax, vmap
from ml_collections import config_dict, config_flags

from model import AutoEncoder
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)

warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)


AUTOTUNE = tf.data.AUTOTUNE
START = int(time.time())
CONFIG = config_flags.DEFINE_config_file('config', 'config.py')

LOGDIR = flags.DEFINE_string('logdir', '/tmp/logdir', '')


def compute_metrics(*, logits: jnp.ndarray, labels: jnp.ndarray) -> dict[str, jnp.ndarray]:
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'accuracy': accuracy,
  }
  return metrics


def accumulate_metrics(metrics: list) -> dict:
  metrics = jax.device_get(metrics)
  return {
    k: np.mean([metric[k] for metric in metrics])
    for k in metrics[0]
  }

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


def create_dataset(fn: str, cfg: config_dict.ConfigDict) -> tf.data.Dataset:
  ds = tf.data.TFRecordDataset(['data/mega-2400-00000-of-00100'], 'ZLIB')
  ds = ds.batch(cfg.train.batch_size, drop_remainder=True)
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES),
              num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  if cfg.train.shuffle:
    ds = ds.shuffle(cfg.train.shuffle)
  ds = ds.as_numpy_iterator()
  return ds



def main(argv):
  try:
    os.mkdir(LOGDIR.value)
  except:
    pass

  cfg = CONFIG.value
  with open(os.path.join(LOGDIR.value, 'config.txt'), 'w') as f:
            f.write(str(cfg))

  rng = jax.random.PRNGKey(int(time.time()))
  rng, rnd_ae = jax.random.split(rng, num=2)
  model = AutoEncoder(**cfg.model)
  x = jax.random.randint(key=rng,
                                shape=((1,) + TRANSFORMER_SHAPE),
                                minval=0,
                                maxval=TRANSFORMER_VOCABULARY,
                                dtype=jnp.int32)


  variables = model.init(rnd_ae, x)
  with open(os.path.join(LOGDIR.value, 'model-tabulate.txt'), 'w') as f:
    f.write(model.tabulate(rng, x, console_kwargs={'width': 120}))
    flattened, _ = jax.tree_util.tree_flatten_with_path(variables)
    for key_path, value in flattened:
      f.write(f'{jax.tree_util.keystr(key_path):40s} {str(value.shape):20s} {str(value.dtype):10s}\n')

  jax.tree_map(lambda x: x.shape, variables)

  optimizer = optax.adamw(learning_rate=0.001)

  ds = create_dataset('data/mega-2400-00000-of-00100', cfg)

  state = train_state.TrainState.create(
    apply_fn=model.apply,
    tx=optimizer,
    params=variables['params'])

  nb = 0
  print('training')
  ar = []

  for batch in iter(ds):
    nb += 1
    label = batch['label']
    label = label.reshape([label.shape[0], 1])
    state, metrics = train_step(state, batch['board'], label)
    print('metrics: ', metrics)
    ar.append(metrics)
    if nb % 100 == 0:
      m = accumulate_metrics(ar)
      print(nb, 'acc: ', jnp.asarray(m['accuracy']), 'loss: ', jnp.asarray(m['loss']))
      ar = []


  print('done')
  print(nb)


if __name__ == "__main__":
  app.run(main)
