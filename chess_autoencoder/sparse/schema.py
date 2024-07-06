try:
  # Get code to work in a pytorch venv.
  import tensorflow as tf
except ImportError:
  pass


# From ChessAtAGlance/encode.py

TRANSFORMER_LENGTH = (64 + 4 + 1)
TRANSFORMER_SHAPE = (TRANSFORMER_LENGTH,)
TRANSFORMER_VOCABULARY = 38
LABEL_VOCABULARY = 1968  # encode_moves.py, all_moves.txt


try:
  TRANSFORMER_FEATURES = {
    'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
  }

  TRANSFORMER_BOARD_FEATURES = {
    'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
  }

  TRANSFORMER_FEATURES_FEN = {
    'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64),
    'fen': tf.io.FixedLenFeature([1], tf.string),
  }
except NameError:
  pass
