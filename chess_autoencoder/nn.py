import json
import sys
import os
import heapq

import tensorflow as tf


import sqlitedict
import numpy as np

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
h = []

tot = len(db)

xtot = None
xtotn = 0
for row, (fen, embedding) in enumerate(db.items()):
  # if xfen is None:
  #   xfen = fen
  #   xembedding = embedding
  #   xembedding = np.array(xembedding)
  #   print(xfen)
  #   continue

  embedding = np.array(embedding)
  if xtot is None:
    xtot = embedding
  else:
    xtot += embedding
  print('CUR: ', type(embedding), embedding)
  xtotn += 1
  if xtotn == 10:
    print('AVG: ', xtot / xtotn)
    break
  dist = np.linalg.norm(embedding - xembedding)

  heapq.heappush(h, (dist, fen))
  if xbest is None or dist < xbest:
    print('best: ', fen, dist)
    xbest = dist
  if xworst is None or dist > xworst:
    print('worst: ', fen, dist)
    xworst = dist
  if row % 5000 == 0:
    print(row, f'{100.0 * row / tot:.1f}%')

print()
print('\n'.join(str(foo) for foo in heapq.nlargest(10, h)))
print()
print('\n'.join(str(foo) for foo in heapq.nsmallest(10, h)))
print()
