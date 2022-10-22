import tensorflow as tf
import json
import sqlitedict
import numpy as np
import sys
import os
import chess
from encode import encode_board, SHAPE, FLAT_SHAPE


FEN = 'r3r1k1/ppq2pbp/2pp1np1/5b2/2PP4/BPN3P1/P1Q2PBP/3RR1K1 w'
FEN = 'rnbqk2r/pp2ppb1/2p4p/6pP/3PN3/8/PPPQ1PP1/2KR1BNR w'

encoder = tf.keras.models.load_model('encoder.model')

board = chess.Board(FEN)
encoded = encode_board(board)
batch = tf.expand_dims(encoded, axis=0)
xembedding = encoder(batch)[0]
print(xembedding)


db = sqlitedict.open(filename='embeddings.sqlite',
                     flag='r',
                     encode=json.dumps,
                     decode=json.loads)

xfen = FEN
# xembedding = None
xbest = None
xworst = None

tot = len(db)
for row, (fen, embedding) in enumerate(db.items()):
  # if xfen is None:
  #   xfen = fen
  #   xembedding = embedding
  #   xembedding = np.array(xembedding)
  #   print(xfen)
  #   continue

  embedding = np.array(embedding)
  dist = np.linalg.norm(embedding - xembedding)

  if xbest is None or dist < xbest:
    print('best: ', fen, dist)
    xbest = dist
  if xworst is None or dist > xworst:
    print('worst: ', fen, dist)
    xworst = dist
  if row % 5000 == 0:
    print(row, f'{100.0 * row / tot:.1f}%')
