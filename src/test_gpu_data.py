"""Sprint 4 check: the torch/GPU unpacker must give exactly the same planes as encode.unpack.
Run this on the training machine before training:  python src/test_gpu_data.py
"""
import random
from pathlib import Path

import chess
import numpy as np
import torch

from encode import compact, unpack
from gpu_data import GpuSplit, unpack_gpu
from moves import move_to_index


def compact_arrays(boards):
    rows = [compact(b) for b in boards]
    return (np.array([r[0] for r in rows], dtype=np.uint64), np.array([r[1] for r in rows], dtype=np.uint8),
            np.array([r[2] for r in rows], dtype=np.int8), np.array([r[3] for r in rows], dtype=np.uint8))


def test_unpack_matches_numpy(device):
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",    # en passant, Black to move
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",    # en passant, White to move
        "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",                              # partial castling rights
    ]
    random.seed(7)
    boards = [chess.Board(f) for f in fens]
    for _ in range(150):                                                    # plus random-play positions
        b = chess.Board()
        for _ in range(random.randint(1, 120)):
            if b.is_game_over():
                break
            b.push(random.choice(list(b.legal_moves)))
            boards.append(chess.Board(b.fen()))
    bb, castle, ep, wtm = compact_arrays(boards)
    expected = unpack(bb, castle, ep, wtm)
    got = unpack_gpu(torch.from_numpy(bb.view(np.int64)).to(device), torch.from_numpy(castle).to(device),
                     torch.from_numpy(ep).to(device), torch.from_numpy(wtm).to(device)).cpu().numpy()
    assert got.shape == expected.shape
    assert np.array_equal(got, expected), "GPU unpack differs from numpy unpack"
    print(f"ok  unpack_gpu == numpy unpack on {len(boards)} positions (device: {device}, "
          f"{int((ep >= 0).sum())} with en passant)")


def test_split_on_device(device, rows=5000):
    path = Path("data/encoded/val.npz")
    if not path.exists():
        print("skip real-data check (data/encoded/val.npz not found)")
        return
    d = np.load(path)
    split = GpuSplit(str(path), device, limit=rows)
    ids = torch.arange(len(split), device=device)
    x, y = split.get(ids)
    expected = unpack(d["bb"][:rows], d["castle"][:rows], d["ep"][:rows], d["wtm"][:rows])
    assert np.array_equal(x.cpu().numpy(), expected)
    assert np.array_equal(y.cpu().numpy(), move_to_index(d["mv"][:rows]))
    print(f"ok  GpuSplit batches and labels match on {len(split)} real rows ({split.nbytes() / 1e6:.1f} MB on {device})")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_unpack_matches_numpy(dev)
    test_split_on_device(dev)
    print("\nALL TESTS PASSED")
