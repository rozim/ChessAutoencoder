import datetime
import functools
import glob
import os
import pprint
import random
import sys
import time
import warnings
from typing import Any

import flax
import jax
import jax.numpy as jnp
import jax.random
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
from jax import grad, jax, jit, lax, vmap
from ml_collections import config_dict, config_flags

from model import AutoEncoderLabelHead
from schema import (LABEL_VOCABULARY, TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY, TRANSFORMER_BOARD_FEATURES)

warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)


AUTOTUNE = tf.data.AUTOTUNE
START = time.time()
CONFIG = config_flags.DEFINE_config_file('config', 'config.py')

LOGDIR = flags.DEFINE_string('logdir', '/tmp/logdir', '')
LABEL_BIAS = flags.DEFINE_string('label_bias', None, 'Text file of label frequencies')

ALL_ACC = flags.DEFINE_boolean('all_accuracy', True, '')


def write_metrics(writer: tf.summary.SummaryWriter,
                  step: int,
                  metrics: Any,
                  hparams: Any = None) -> None:

  with writer.as_default(step):
    for k, v in metrics.items():
      tf.summary.scalar(k, v)
    if hparams:
      hp.hparams(hparams)
    if True: # acc hack
      hv = []
      for k in sorted(metrics.keys()):
        if 'x_acc/acc_' in k: # acc/acc#
          hv.append(metrics[k])
      assert len(hv)
      tf.summary.histogram('x_acc_hist', hv, buckets=TRANSFORMER_LENGTH)

  writer.flush()

def compute_metrics(*, logits: jnp.ndarray, labels: jnp.ndarray) -> dict[str, jnp.ndarray]:
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)


  metrics = {
    'accuracy': accuracy,
  }
  if ALL_ACC.value:
    for i in range(TRANSFORMER_LENGTH):
      metrics[f'x_acc/acc_{i:02d}'] = jnp.mean(jnp.argmax(logits[:, :, i:(i+1)], -1) == labels[:, i:(i+1)])
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

  if cfg.label == 'move':
    features = TRANSFORMER_FEATURES
  else:
    features = TRANSFORMER_BOARD_FEATURES

  ds = ds.map(functools.partial(tf.io.parse_example, features=features),
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
  # datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

  try:
    os.mkdir(LOGDIR.value)
  except:
    pass

  cfg = CONFIG.value
  assert cfg.label in ['move', 'board']
  model_extra = {}
  if cfg.label == 'board':
    move_extra = {'label_vocabulary': TRANSFORMER_VOCABULARY}
  log_config(cfg)

  rng = jax.random.PRNGKey(int(time.time()))
  rng, rnd_ae = jax.random.split(rng, num=2)

  model = AutoEncoderLabelHead(**cfg.model)
  x = jax.random.randint(key=rng,
                                shape=((1,) + TRANSFORMER_SHAPE),
                                minval=0,
                                maxval=TRANSFORMER_VOCABULARY,
                                dtype=jnp.int32)


  variables = model.init(rnd_ae, x)
  with open(os.path.join(LOGDIR.value, 'model-tabulate.txt'), 'w') as f:
    f.write(model.tabulate(rng, x))
    #table_kwargs={'width': 180}))
    #console_kwargs={'width': 180},
    #column_kwargs={'width': 180}))
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

    if cfg.label == 'move':
      label = batch['label']
      label = label.reshape([label.shape[0], 1])
    else:
      label = batch['board']

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

      sw_m = {'accuracy': train_acc,
              'loss': train_loss,
              'time/elapsed': dt,
              'time/xps': (1000 * cfg.train.batch_size) / dt}
      if ALL_ACC.value:
        for k, v in m.items():
          if 'acc/acc' in k:
            sw_m[k] = v
      write_metrics(train_writer, step, sw_m)

      ar = []

  c_mngr.wait_until_finished()  # async..
  csv.close()
  csv = None
  tsv.close()
  tsv = None

  print('all done')


if __name__ == "__main__":
  app.run(main)
