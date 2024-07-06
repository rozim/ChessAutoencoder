# Study labels, print bias

import collections
import functools
import glob

import jax
import optax
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from jax import grad
from jax import numpy as jnp


@jax.jit
def loss_fn(logits, label):
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
  return loss.mean()

def main(argv):
  lr = 0.001
  files = glob.glob('data/mega-2400-0000?-of-00100')
  print(f'NF: {len(files)}')
  ds = tf.data.TFRecordDataset(files, 'ZLIB')
  ds = ds.map(functools.partial(tf.io.parse_example, features={
    'label': tf.io.FixedLenFeature([1], tf.int64)}))
  #ds = ds.take(10000)
  ds = ds.as_numpy_iterator()

  bias = jnp.zeros(1968)

  with open('study0.txt', 'w') as f:
    for foo in jax.nn.softmax(bias):
      f.write(f'{foo:.6f}\n')

  grad_fn = jax.grad(loss_fn)
  #for batch in tqdm.tqdm(ds):
  d = collections.Counter()
  print('reading')
  for nr, batch in enumerate(ds):
    label = batch['label']
    d[label] += 1
    grad = grad_fn(bias, label)
    bias = bias - grad * lr
  print(f'reading done nr={nr}')

  with open('study-manual.txt', 'w') as f:
    tot = sum(d.values())
    for i in range(1968):
      n = d.get(i, 0)
      f.write(f'{i:4d} {n/tot:.6f} {n:8d}\n')

  with open('study-grad.txt', 'w') as f:
    for g in grad:
      f.write(f'{g:.6f}\n')
  with open('study.txt', 'w') as f:
    best = -1
    besti = -1
    for i, foo in enumerate(jax.nn.softmax(bias)):
      f.write(f'{foo:.6f}\n')
      if foo > best:
        best = foo
        besti = i
  print(f'best: {besti}: {best}')




if __name__ == "__main__":
  app.run(main)
