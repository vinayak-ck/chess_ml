"""Sprint 3: fixed mapping between moves and policy-network output indices.

All moves live in the ORIENTED frame (side to move = White, see encode.py).

Index layout (NUM_MOVES = 4168):
  0 .. 4095     from_square * 64 + to_square      (all normal moves, castling,
                                                    en passant, and QUEEN promotions)
  4096 .. 4167  under-promotions (knight / bishop / rook):
                4096 + ((piece - 2) * 8 + from_file) * 3 + (to_file - from_file + 1)
"""
import chess
import numpy as np

NUM_MOVES = 4096 + 3 * 8 * 3   # 4168


def move_to_index(mv: np.ndarray) -> np.ndarray:
    """Vectorised. mv: (N, 3) array of (from_square, to_square, promotion; 0 = none)."""
    mv = np.asarray(mv).astype(np.int64)
    frm, to, promo = mv[:, 0], mv[:, 1], mv[:, 2]
    plain = frm * 64 + to
    under = (promo >= chess.KNIGHT) & (promo <= chess.ROOK)       # 2, 3, 4
    from_file = frm % 8
    delta = (to % 8) - from_file + 1                               # 0, 1 or 2
    under_idx = 4096 + ((promo - 2) * 8 + from_file) * 3 + delta
    return np.where(under, under_idx, plain)


def move_index(move: chess.Move) -> int:
    """One chess.Move (already oriented) -> index."""
    return int(move_to_index(np.array([[move.from_square, move.to_square,
                                        move.promotion or 0]]))[0])


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Index -> chess.Move on the ORIENTED board (side to move = White)."""
    if idx < 4096:
        frm, to = divmod(idx, 64)
        promo = None
        piece = board.piece_at(frm)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to) == 7:
            promo = chess.QUEEN            # plain index on the last rank = queen promotion
        return chess.Move(frm, to, promo)
    k = idx - 4096
    delta = k % 3 - 1
    k //= 3
    from_file = k % 8
    promo = k // 8 + 2
    frm = 6 * 8 + from_file               # rank 7 (index 6)
    to = 7 * 8 + from_file + delta        # rank 8 (index 7)
    return chess.Move(frm, to, promo)


def legal_move_mask(oriented_board: chess.Board) -> np.ndarray:
    """Boolean (NUM_MOVES,) mask of the legal moves in an oriented position."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for m in oriented_board.legal_moves:
        mask[move_index(m)] = True
    return mask