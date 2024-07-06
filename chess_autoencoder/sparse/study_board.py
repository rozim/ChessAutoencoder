# Study board input distribution.

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

from schema import (TRANSFORMER_FEATURES, TRANSFORMER_LENGTH,
                    TRANSFORMER_SHAPE, TRANSFORMER_VOCABULARY)

def main(argv):
  lr = 0.001
  files = glob.glob('data/mega-2400-000??-of-00100')
  print(f'NF: {len(files)}')
  ds = tf.data.TFRecordDataset(files, 'ZLIB')
  ds = ds.map(functools.partial(tf.io.parse_example, features={
    'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64)}))

  #ds = ds.take(100000)

  ds = ds.as_numpy_iterator()

  data = [collections.Counter() for _ in range(TRANSFORMER_LENGTH)]
  print('reading')

  for nr, batch in enumerate(ds):
    board = batch['board']
    for i in range(TRANSFORMER_LENGTH):
      data[i][board[i]] += 1

  ar = []
  for i in range(TRANSFORMER_LENGTH):
    ar.append(f'{i:3d}')
  print(f'  : {" ".join(ar)}')
  for i in range(TRANSFORMER_LENGTH):
    ar = []
    for j in range(TRANSFORMER_LENGTH):
      if data[i][j] == 0:
        ar.append(' . ')
      else:
        ar.append(f'{data[i][j]:3d}')
    print(f'{i:02d}: {" ".join(ar)}')
  print()
  print('# Details')
  print()
  for i in range(TRANSFORMER_LENGTH):
    print()
    print('loc: ', i)
    tot = float(sum(data[i].values()))
    for j in range(TRANSFORMER_LENGTH):
      res = data[i][j]
      if res  > 0:
        print(f'\t{j:2d} {res:10d} {(res/tot)*100.0:5.1f}%')






if __name__ == "__main__":
  app.run(main)


#   TRANSFORMER_CO_I2P = {
#   1: (BLACK, PAWN),
#   2: (BLACK, KNIGHT),
#   3: (BLACK, BISHOP),
#   4: (BLACK, ROOK),
#   5: (BLACK, QUEEN),
#   6: (BLACK, KING),

#   7: (WHITE, PAWN),
#   8: (WHITE, KNIGHT),
#   9: (WHITE, BISHOP),
#   10: (WHITE, ROOK),
#   11: (WHITE, QUEEN),
#   12: (WHITE, KING),
# }
