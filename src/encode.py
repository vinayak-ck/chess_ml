"""Sprint 2: board -> tensor encoding, from the side-to-move's point of view."""
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]
NUM_PLANES = 18
# Plane layout (channels-first, each plane is 8x8 indexed [rank][file]):
#   0-5    side-to-move's  P N B R Q K
#   6-11   opponent's      P N B R Q K
#   12     side-to-move can castle kingside     13  ... queenside
#   14     opponent can castle kingside         15  ... queenside
#   16     en-passant target square (if a legal ep capture exists)
#   17     all ones if White is to move, all zeros if Black


def orient(board: chess.Board) -> chess.Board:
    """Return a position where the side to move is 'White'.
    If Black is to move, flip the board vertically and swap colours."""
    return board if board.turn == chess.WHITE else board.mirror()


def orient_move(move: chess.Move, turn: bool) -> chess.Move:
    """Apply the same flip to a move. `turn` is board.turn BEFORE the move.
    The flip is its own inverse, so this also maps back."""
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)


def compact(board: chess.Board):
    """Board -> (12 bitboards, castling bits, ep square or -1, white_to_move)."""
    white_to_move = int(board.turn == chess.WHITE)
    b = orient(board)
    bbs = [b.pieces_mask(pt, chess.WHITE) for pt in PIECE_TYPES]
    bbs += [b.pieces_mask(pt, chess.BLACK) for pt in PIECE_TYPES]
    castle = (int(b.has_kingside_castling_rights(chess.WHITE))
              + 2 * int(b.has_queenside_castling_rights(chess.WHITE))
              + 4 * int(b.has_kingside_castling_rights(chess.BLACK))
              + 8 * int(b.has_queenside_castling_rights(chess.BLACK)))
    ep = -1 if b.ep_square is None else b.ep_square
    return bbs, castle, ep, white_to_move


def unpack(bbs, castle, ep, white_to_move):
    """Vectorised: compact arrays -> float32 tensor of shape (B, 18, 8, 8).
    bbs (B,12) uint64 | castle (B,) uint8 | ep (B,) int8 | white_to_move (B,) uint8
    """
    B = len(bbs)
    out = np.zeros((B, NUM_PLANES, 8, 8), dtype=np.float32)

    raw = np.ascontiguousarray(bbs, dtype="<u8").view(np.uint8)       # (B, 96)
    bits = np.unpackbits(raw, axis=-1, bitorder="little")              # (B, 768)
    out[:, :12] = bits.reshape(B, 12, 8, 8)    # bit i = square i = rank*8 + file

    castle = np.asarray(castle)
    for k in range(4):
        out[:, 12 + k] = ((castle >> k) & 1)[:, None, None]

    ep = np.asarray(ep).astype(np.int64)
    idx = np.nonzero(ep >= 0)[0]
    out[idx, 16, ep[idx] // 8, ep[idx] % 8] = 1.0

    out[:, 17] = np.asarray(white_to_move)[:, None, None]
    return out


def encode_board(board: chess.Board) -> np.ndarray:
    """Convenience: one board -> (18, 8, 8) float32."""
    bbs, castle, ep, wtm = compact(board)
    return unpack(np.array([bbs], dtype=np.uint64), np.array([castle], dtype=np.uint8),
                  np.array([ep], dtype=np.int8), np.array([wtm], dtype=np.uint8))[0]


def show_plane(x: np.ndarray, plane: int) -> None:
    """Print one plane with rank 8 at the top, like a real board."""
    for rank in range(7, -1, -1):
        print(rank + 1, " ".join(str(int(v)) for v in x[plane, rank]))
    print("  a b c d e f g h")