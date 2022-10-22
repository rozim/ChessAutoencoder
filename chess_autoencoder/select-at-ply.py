import sys
import os
import random

import chess
import chess.pgn

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('ply', 40, '')


def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13

  # We only care about pieces and side to move - downstream the autoencoder
  # ignores castling etc.
  return ' '.join(board.fen().split(' ')[0:2])

def gen_games(fn):
  with open(fn, 'r', encoding='utf-8', errors='replace') as f:
    while True:
      g = chess.pgn.read_game(f)
      if g is None:
        return
      yield g


def find_pos_at_ply(game, target_ply):
  board = game.board()
  for game_ply, move in enumerate(game.mainline_moves()):
    if game_ply == target_ply:
      return simplify_fen(board)
    board.push(move)



def main(argv):
  all = set()
  for fn in argv[1:]:
    for game in gen_games(fn):
      sfen = find_pos_at_ply(game, FLAGS.ply)
      if sfen is not None:
        all.add(sfen)
  flat = list(all)
  random.shuffle(flat)
  print('\n'.join(flat))


if __name__ == "__main__":
  app.run(main)
