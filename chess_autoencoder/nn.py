import json
import sqlitedict
import numpy as np
import sys
import os

db = sqlitedict.open(filename='embeddings.sqlite',
                     flag='r',
                     encode=json.dumps,
                     decode=json.loads)

xfen = None
xembedding = None
xbest = None
xworst = None

for fen, embedding in db.items():
  if xfen is None:
    xfen = fen
    xembedding = embedding
    xembedding = np.array(xembedding)
    print(xfen)
    continue

  embedding = np.array(embedding)
  dist = np.linalg.norm(embedding - xembedding)

  if xbest is None or dist < xbest:
    print('best: ', fen, dist)
    xbest = dist
  if xworst is None or dist > xworst:
    print('worst: ', fen, dist)
    xworst = dist
