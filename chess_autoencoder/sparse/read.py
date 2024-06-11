import tensorflow as tf
import time
import functools
import sys, os

from absl import app

from schema import TRANSFORMER_FEATURES_FEN

def main(argv):
  ds = tf.data.TFRecordDataset(['data/mega-2400-00000-of-00100'], 'ZLIB')
  ds = ds.batch(1)
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_FEATURES_FEN))
  for ent in iter(ds):
    print('board', ent['board'])
    print('label', ent['label'])
    print('fen', ent['fen'])
    break



if __name__ == '__main__':
  app.run(main)
