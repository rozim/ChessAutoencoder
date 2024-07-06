/Users/dave/Projects/ChessAtAGlance
source ~/venv-jax/bin/activate

# Generate training data
nice time python generate_transformer_training_data.py --pgn=/Users/dave/Projects/ChessData/Release/2023-12-14/mega-clean-2400.pgn  -out=/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400 --shards=100

# Generate training data into sqlite

nice time python generate_transformer_training_data.py --pgn=/Users/dave/Projects/ChessData/Release/2023-12-14/mega-clean-2400.pgn  -sqlite_out=/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.db
...
COMMIT 51000000
806912 67162236 51007106 16155130
807936 67212408 51041981 16170427
n_game:  808201
n_move:  67224915
n_dup:  16174220
n_gen:  51050695
    16588.30 real      7287.68 user      9727.34 sys

ls -lh /Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.db
-rw-r--r--  1 dave  staff    10G Jul  3 22:29 /Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.db

>>> import sqlitedict
>>> db = sqlitedict.open('/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.db', flag='r')
>>> len(db)
48753958

# Convert from db to *.jsonl.gz file (30mins)
time python db2jsonlines.py

# Best effort shuffle (2h!)
time python jsonlines_semi_shuffle.py
...
48700000 47700000 6988
     7146.40 real      5812.01 user        85.58 sys

run jsonlines_split.py to spit into train/test


# Schema
Schema: from ChessAtAGlance/encode.py
# 64: squares
# 4: castle
# 1: ep square
TRANSFORMER_LENGTH = (64 + 4 + 1)
TRANSFORMER_SHAPE = (TRANSFORMER_LENGTH,)
TRANSFORMER_VOCABULARY = 38

encode_move.py: 1968 moves

Schema: from generate_transformer_training_data.py:
        feature = {
          'board': int64_feature_alt(enc_board),	# 69,38
          'label': int64_feature(enc_move),		# 1968
          'fen':  bytes_feature(fen.encode('utf-8')),
        }

# Print 1 record

python read.py
board tf.Tensor(
[[10  0  0 11 12  0  0 10  0  7  0  0  0  0  7  0  7  0  8  7  9  8  0  7
   0  0  0  7  7  7  0  0  0  0  1  0  0  0  0  0  0  1  0  1  0  2  1  0
   1  0  0  0  1  1  3  1  4  0  3  5  0  4  6  0 13 15 18 20 31]], shape=(1, 69), dtype=int64)
label tf.Tensor([724], shape=(1,), dtype=int64)
fen tf.Tensor([b'r1bq1rk1/p3ppbp/1p1p1np1/2p5/3PPP2/P1NPBN1P/1P4P1/R2QK2R w KQ -'], shape=(1,), dtype=string)
