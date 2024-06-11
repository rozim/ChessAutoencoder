/Users/dave/Projects/ChessAtAGlance
source ~/venv-jax/bin/activate

# Generate training data
nice time python generate_transformer_training_data.py --pgn=/Users/dave/Projects/ChessData/Release/2023-12-14/mega-clean-2400.pgn  -out=/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400 --shards=100

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
