import json
import sys
import os
import heapq

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, Accuracy, BinaryAccuracy, MeanSquaredError

import chess
from encode import encode_board, SHAPE, FLAT_SHAPE

encoder = tf.keras.models.load_model('encoder-foo.model')
autoencoder = tf.keras.models.load_model('autoencoder-foo.model')


FEN = 'r1b1kb1r/2p1qppp/p2p4/8/3Nn3/8/PPP1QPPP/RNB2RK1 w'
board = chess.Board(FEN)
encoded = encode_board(board)
batch = tf.expand_dims(encoded, axis=0)
print('BATCH: ', batch)
xembedding = encoder(batch)[0]
print('EMBEDDING: ', xembedding)
print()
x = tf.expand_dims(encoded, axis=0)
fx = tf.reshape(x, (1, FLAT_SHAPE))
y = autoencoder(x)
print('INFERENCE: ', y)

loss_fn = BinaryCrossentropy()
bin_acc_tracker = BinaryAccuracy()
mse_tracker = MeanSquaredError()

loss = loss_fn(y, fx)
print('LOSS: ', loss)

bin_acc_tracker.update_state(y_true=fx, y_pred=y)
mse_tracker.update_state(y_true=fx, y_pred=y)
print('BIN ACC: ', bin_acc_tracker.result().numpy())
print('MSE: ',     mse_tracker.result().numpy())

print(fx.shape)
print('nz: ', tf.math.count_nonzero(fx))
print('nz: ', tf.math.count_nonzero(y))
yy = tf.cast(y > 0.5, tf.float32)
print('nz: ', tf.math.count_nonzero(yy))
print()
print('fx where: ', tf.where(tf.squeeze(fx)))
print('yy where: ', tf.where(tf.squeeze(yy)))
# print('yy where: ', tf.where(tf.squeeze(tf.squeeze(yy), axis=-1)))
