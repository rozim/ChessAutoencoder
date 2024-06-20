import functools
import random
import sys
import glob
import os
import time
import warnings
from typing import Any
import pprint

import flax
import jax
import jax.numpy as jnp
import jax.random
from jax import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import app, flags, logging
from clu import metrics
from flax import jax_utils
from flax import linen as nn
from flax import struct
from flax.training import common_utils, train_state
from jax import grad, jit, lax, vmap
from ml_collections import config_dict, config_flags

from model import AutoEncoderLabelHead
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)

warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)


AUTOTUNE = tf.data.AUTOTUNE
START = time.time()
CONFIG = config_flags.DEFINE_config_file('config', 'config.py')

LOGDIR = flags.DEFINE_string('logdir', '/tmp/logdir', '')
LABEL_BIAS = flags.DEFINE_string('label_bias', None, 'Text file of label frequencies')


def write_metrics(writer: tf.summary.SummaryWriter,
                  step: int,
                  metrics: Any,
                  hparams: Any = None) -> None:
  with writer.as_default(step):
    for k, v in metrics.items():
      tf.summary.scalar(k, v)
    if hparams:
      hp.hparams(hparams)
  writer.flush()

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


def create_dataset(pat: str, cfg: config_dict.ConfigDict) -> tf.data.Dataset:
  files = glob.glob(pat)
  assert len(files) > 0, [pat, glob.glob(pat)]
  random.shuffle(files)

  ds = tf.data.TFRecordDataset(files, 'ZLIB', num_parallel_reads=4)

  if cfg.train.shuffle:
    ds = ds.shuffle(cfg.train.shuffle)

  ds = ds.batch(cfg.train.batch_size, drop_remainder=True)

  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES),
              num_parallel_calls=AUTOTUNE, deterministic=False)

  ds = ds.prefetch(AUTOTUNE)

  ds = ds.as_numpy_iterator()
  return ds


def log_config(cfg: config_dict.ConfigDict):
  with open(os.path.join(LOGDIR.value, 'config.txt'), 'w') as f:
    f.write(pprint.pformat(cfg))
  with open(os.path.join(LOGDIR.value, 'config.json'), 'w') as f:
    f.write(cfg.to_json())


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  try:
    os.mkdir(LOGDIR.value)
  except:
    pass

  cfg = CONFIG.value
  log_config(cfg)

  label_frequencies = None
  if LABEL_BIAS.value:
    with open(LABEL_BIAS.value, 'r') as f:
      label_frequencies = jnp.asarray([float(foo) for foo in f.readlines()])
      #ar += 1e-10 # avoid division by 0
      #bias_init = jnp.log(1.0 / ar)
      #assert jnp.all(lax.is_finite(bias_init))


  rng = jax.random.PRNGKey(int(time.time()))
  rng, rnd_ae = jax.random.split(rng, num=2)

  if label_frequencies is not None:
    hack = config_dict.ConfigDict(cfg.model)
    hack.unlock()
    hack.label_frequencies = label_frequencies
  else:
    hack = cfg.model
  model = AutoEncoderLabelHead(**hack)
  #model = AutoEncoderLabelHead(**cfg.model)
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
  #optimizer = optax.chain(
  #optax.clip(1.0),  # Clip gradients at 1
  #optax.adam(lr_schedule)
  #)

  ds = create_dataset('data/mega-2400-?????-of-?????', cfg)

  state = train_state.TrainState.create(
    apply_fn=model.apply,
    tx=optimizer,
    params=variables['params'])


  c_path = ocp.test_utils.erase_and_create_empty(os.path.join(LOGDIR.value, 'checkpoint'))
  c_options = ocp.CheckpointManagerOptions(max_to_keep=3)
  c_mngr = ocp.CheckpointManager(c_path, options=c_options)

  step = 0
  print('training')
  ar = []

  csv = open(os.path.join(LOGDIR.value, 'train.csv'), 'w')
  csv.write('step,elapsed,loss,accuracy\n')
  tsv = open(os.path.join(LOGDIR.value, 'train.tsv'), 'w')
  tsv.write('step\telapsed\tloss\taccuracy\n')
  tsv.flush()

  train_writer = tf.summary.create_file_writer(
    os.path.join(LOGDIR.value, 'train'))

  t1 = time.time()
  for batch in iter(ds):
    step += 1
    label = batch['label']
    label = label.reshape([label.shape[0], 1])
    state, metrics = train_step(state, batch['board'], label)
    # print('metrics: ', metrics)
    ar.append(metrics)
    if step % 1000 == 0:
      c_mngr.save(step, args=ocp.args.StandardSave(state))
      m = accumulate_metrics(ar)
      train_loss = jnp.asarray(m['loss'])
      train_acc = jnp.asarray(m['accuracy'])
      elapsed = time.time() - START
      dt = time.time() - t1
      t1 = time.time()
      print(f'{step:6d} dt={elapsed:6.1f}, loss={train_loss:6.4f}, acc={train_acc:6.4f}')
      csv.write(f'{step},{elapsed},{train_loss},{train_acc}\n')
      csv.flush()
      tsv.write(f'{step:12d}\t{elapsed:12f}\t{train_loss:12f}\t{train_acc:12f}\n')
      tsv.flush()

      write_metrics(train_writer, step,
                    {'accuracy': train_acc,
                     'loss': train_loss,
                     'time/elapsed': dt,
                     'time/xps': (1000 * cfg.train.batch_size) / dt})
      ar = []

  c_mngr.wait_until_finished()  # async..
  csv.close()
  csv = None
  tsv.close()
  tsv = None

  print('all done')


if __name__ == "__main__":
  app.run(main)
