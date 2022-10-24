import sys
import functools

from encode import encode_board, SHAPE, FLAT_SHAPE, SHAPE_2D
import chess

from absl import app
from absl import flags

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Reshape, Input, Dense, Reshape, Activation
from tensorflow.keras.layers import Permute, Conv2D, LayerNormalization, Conv2DTranspose
from tensorflow.keras.layers import Add


def create_models(encoding_dim=32):
  input_img = Input(shape=SHAPE)
  flat = Flatten()(input_img)
  encoded = Dense(encoding_dim, activation='relu')(flat)
  #unflat = Reshape(SHAPE)(encoded)
  decoded = Dense(FLAT_SHAPE, activation='sigmoid')(encoded)

  encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
  autoencoder = Model(inputs=input_img, outputs=decoded, name='autoencoder')
  return autoencoder, encoder

def create_conv_models(encoding_dim=32,
                       num_layers=1,
                       num_filters=10,
                       act_fn = 'relu',
                       kernel='glorot_uniform'):
  input_img = Input(shape=SHAPE)

  x = Reshape(SHAPE_2D)(input_img)
  # shape: 12,       8, 8
  #        channels, x, y
  #
  x = Permute([2, 3, 1])(x)
  # shape: 8, 8, 12
  data_format = 'channels_last'

  my_conv2d = functools.partial(Conv2D,
                                filters=num_filters,
                                kernel_size=(3, 3),
                                data_format=data_format,
                                kernel_initializer=kernel,
                                padding='same',
                                use_bias=False)

  x = my_conv2d()(x)

  for i in range(num_layers):
    skip = x
    x = my_conv2d()(x)
    x = LayerNormalization()(x)
    x = Activation(act_fn)(x)

    x = my_conv2d()(x)
    x = LayerNormalization()(x)
    x = Add()([x, skip])
    x = Activation(act_fn)(x)

  flat = Flatten()(x)
  # tanh -> embeddings should be 0..1
  encoded = Dense(encoding_dim, kernel_initializer=kernel, activation='tanh')(flat)

  x = encoded
  x = Dense(64, kernel_initializer=kernel)(x)
  x = Reshape((1, 8, 8))(x)
  x = Permute([2, 3, 1])(x)
  for _ in range(num_layers):
    x = Conv2DTranspose(num_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=kernel)(x)
    x = LayerNormalization()(x)
    x = Activation(act_fn)(x)

  x = Flatten()(x)
  # sigmoid -> we know the input is in 0..1
  decoded = Dense(FLAT_SHAPE, activation='sigmoid', kernel_initializer=kernel)(x)

  encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
  autoencoder = Model(inputs=input_img, outputs=decoded, name='autoencoder')

  # tbd: more exact output, this may do
  # tf.multiply(rows, tf.cast(tf.equal(rows, tf.reduce_max(rows, axis=0)), tf.int32))
  return autoencoder, encoder


def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  if False:
    board = chess.Board(fen)
    enc = encode_board(board)
    print(enc)
    enc = tf.expand_dims(enc, axis=0)
    print()
    print(enc)


    print()
    print('#')
    #encoder.summary()
    print('#')
    #res = encoder(enc)
    #print('ENC: ', res)

  autoencoder, encoder = create_models()
  autoencoder.summary()
  print()
  print('# conv')
  print()
  autoencoder, encoder = create_conv_models(num_filters=12, num_layers=2)

  autoencoder.summary()
  print()
  print('#')
  encoder.summary()
  print('#')



if __name__ == "__main__":
  app.run(main)
