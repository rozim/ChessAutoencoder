import tensorflow as tf

# From ChessAtAGlance/encode.py

TRANSFORMER_LENGTH = (64 + 4 + 1)
TRANSFORMER_SHAPE = (TRANSFORMER_LENGTH,)
TRANSFORMER_VOCABULARY = 38
LABEL_VOCABULARY = 1968  # encode_moves.py, all_moves.txt

TRANSFORMER_FEATURES = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

TRANSFORMER_BOARD_FEATURES = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
}

TRANSFORMER_FEATURES_FEN = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'fen': tf.io.FixedLenFeature([], tf.string),
}
