"""Sprint 2 tests: fast encoder vs. a slow, obviously-correct reference."""
import random
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from encode import compact, encode_board, orient, orient_move, unpack


def reference(board: chess.Board) -> np.ndarray:
    """Independent slow encoder: loops over squares, flips ranks by hand
    (does NOT use board.mirror(), so it can catch mistakes in the fast path)."""
    me = board.turn
    flip = me == chess.BLACK
    x = np.zeros((18, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r, f = chess.square_rank(sq), chess.square_file(sq)
        if flip:
            r = 7 - r
        offset = 0 if piece.color == me else 6
        x[offset + piece.piece_type - 1, r, f] = 1
    x[12] = board.has_kingside_castling_rights(me)
    x[13] = board.has_queenside_castling_rights(me)
    x[14] = board.has_kingside_castling_rights(not me)
    x[15] = board.has_queenside_castling_rights(not me)
    if board.ep_square is not None:
        r, f = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        x[16, 7 - r if flip else r, f] = 1
    x[17] = float(me == chess.WHITE)
    return x


def batch(boards):
    rows = [compact(b) for b in boards]
    return unpack(np.array([r[0] for r in rows], dtype=np.uint64),
                  np.array([r[1] for r in rows], dtype=np.uint8),
                  np.array([r[2] for r in rows], dtype=np.int8),
                  np.array([r[3] for r in rows], dtype=np.uint8))


def test_start_position():
    x = encode_board(chess.Board())
    assert x.shape == (18, 8, 8)
    assert x[0].sum() == 8 and x[0, 1].sum() == 8          # my pawns on rank 2
    assert x[6].sum() == 8 and x[6, 6].sum() == 8          # their pawns on rank 7
    assert x[:6].sum() == 16 and x[6:12].sum() == 16
    assert (x[12:16] == 1).all() and x[16].sum() == 0 and (x[17] == 1).all()
    print("ok  start position")


def test_black_to_move_is_flipped():
    b = chess.Board()
    b.push_uci("e2e4")                                      # Black to move
    x = encode_board(b)
    assert x[0, 1].sum() == 8                               # Black's pawns now on "rank 2"
    assert x[6, 4, 4] == 1                                  # White's e4 pawn -> rank index 4, file e
    assert (x[17] == 0).all()
    print("ok  black-to-move flip")


def test_en_passant():
    fen = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3"
    b = chess.Board(fen)
    assert b.ep_square == chess.E3
    x = encode_board(b)
    assert x[16].sum() == 1 and x[16, 5, 4] == 1            # e3 flips to rank index 5
    print("ok  en passant")


def test_fuzz_against_reference(n_games=300):
    random.seed(0)
    boards, stats = [], {"black": 0, "ep": 0, "no_castle": 0}
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(random.randint(1, 140)):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
            fb = chess.Board(b.fen())                       # same path as the dataset (FEN)
            boards.append(fb)
            stats["black"] += fb.turn == chess.BLACK
            stats["ep"] += fb.ep_square is not None
            stats["no_castle"] += not fb.castling_rights
    got = batch(boards)
    for i, fb in enumerate(boards):
        assert np.array_equal(got[i], reference(fb)), fb.fen()
    print(f"ok  fuzz: {len(boards)} positions match reference {stats}")


def test_move_orientation(n_games=150):
    random.seed(1)
    n = 0
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(random.randint(1, 100)):
            if b.is_game_over():
                break
            legal = list(b.legal_moves)
            ob = orient(b)
            for m in legal:
                om = orient_move(m, b.turn)
                assert om in ob.legal_moves, (b.fen(), m)
                assert orient_move(om, b.turn) == m          # flip is its own inverse
                n += 1
            b.push(random.choice(legal))
    print(f"ok  move orientation: {n} moves checked")


def test_real_labels(path="data/raw/val.csv.gz", rows=20000):
    if not Path(path).exists():
        print("skip real-data label check (file not found)")
        return
    df = pd.read_csv(path, nrows=rows)
    for fen, uci in zip(df.fen, df.move):
        b = chess.Board(fen)
        assert orient_move(chess.Move.from_uci(uci), b.turn) in orient(b).legal_moves
    print(f"ok  real data: {len(df)} labels are legal after orientation")


if __name__ == "__main__":
    test_start_position()
    test_black_to_move_is_flipped()
    test_en_passant()
    test_fuzz_against_reference()
    test_move_orientation()
    test_real_labels()
    print("\nALL TESTS PASSED")