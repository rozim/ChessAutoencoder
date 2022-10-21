from encode import encode_board, SHAPE, FLAT_SHAPE
import chess

from absl import app
from absl import flags

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Reshape, Input, Dense


def create_models(encoding_dim=32):
  input_img = Input(shape=SHAPE)
  flat = Flatten()(input_img)
  encoded = Dense(encoding_dim, activation='relu')(flat)
  #unflat = Reshape(SHAPE)(encoded)
  decoded = Dense(FLAT_SHAPE, activation='sigmoid')(encoded)

  encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
  autoencoder = Model(inputs=input_img, outputs=decoded, name='autoencoder')
  return autoencoder, encoder


def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)
  enc = encode_board(board)
  print(enc)
  enc = tf.expand_dims(enc, axis=0)
  print()
  print(enc)

  autoencoder, encoder = create_models()

  autoencoder.summary()
  print()
  print('#')
  encoder.summary()
  print('#')
  res = encoder(enc)
  print('ENC: ', res)


if __name__ == "__main__":
  app.run(main)
