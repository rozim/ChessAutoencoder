import tensorflow as tf

# From ChessAtAGlance/encode.py

TRANSFORMER_LENGTH = (64 + 4 + 1)
TRANSFORMER_SHAPE = (TRANSFORMER_LENGTH,)
TRANSFORMER_VOCABULARY = 38

TRANSFORMER_FEATURES = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64)
}
TRANSFORMER_FEATURES_FEN = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'fen': tf.io.FixedLenFeature([], tf.string),
}
