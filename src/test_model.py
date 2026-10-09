"""Sprint 3 tests: move mapping, data loader, model shapes, and an overfit-one-batch check."""
import math
import random
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import EncodedSplit
from encode import orient
from model import build, count_params
from moves import NUM_MOVES, index_to_move, legal_move_mask, move_index


def check_position(fen: str) -> int:
    """Every legal move gets a unique in-range index that decodes back to the same move."""
    b = chess.Board(fen)
    seen = {}
    for m in b.legal_moves:
        i = move_index(m)
        assert 0 <= i < NUM_MOVES, (fen, m, i)
        assert i not in seen, f"collision {m} vs {seen[i]} in {fen}"
        seen[i] = m
        assert index_to_move(i, b) == m, (fen, m, i, index_to_move(i, b))
    return len(seen)


def test_special_positions():
    fens = [
        "8/P6k/8/8/8/8/8/K7 w - - 0 1",                    # promotion on a8
        "1n5k/P7/8/8/8/8/8/K7 w - - 0 1",                   # promotion + capture promotion
        "1n1n3k/2P5/8/8/8/8/8/K7 w - - 0 1",                # c7: b8, c8, d8 all possible
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",            # both castling moves
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",   # en passant
    ]
    counts = [check_position(f) for f in fens]
    # the three-way promotion position must expose 3 targets x 4 pieces = 12 promotion moves
    assert counts[2] >= 12
    print(f"ok  special positions {counts}")


def test_mapping_fuzz(n_games=200):
    random.seed(3)
    n = 0
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(random.randint(1, 150)):
            if b.is_game_over():
                break
            n += check_position(orient(b).fen())
            b.push(random.choice(list(b.legal_moves)))
    print(f"ok  mapping fuzz: {n} legal moves round-trip with no collisions")


def test_real_labels(rows=20000):
    csv, npz = Path("data/raw/val.csv.gz"), Path("data/encoded/val.npz")
    if not (csv.exists() and npz.exists()):
        print("skip real-data label check (files not found)")
        return
    df = pd.read_csv(csv, nrows=rows)
    split = EncodedSplit(str(npz), limit=rows)
    assert len(split) == len(df)
    assert split.y.min() >= 0 and split.y.max() < NUM_MOVES
    for fen, label in zip(df.fen, split.y):
        ob = orient(chess.Board(fen))
        assert index_to_move(int(label), ob) in ob.legal_moves
    print(f"ok  real data: {len(df)} stored labels decode to legal moves "
          f"({len(np.unique(split.y))} distinct classes in this slice)")


def test_model_shapes_and_init_loss():
    for name in ("small", "base"):
        model = build(name)
        x = torch.zeros(4, 18, 8, 8)
        out = model(x)
        assert out.shape == (4, NUM_MOVES)
        print(f"ok  {name}: {count_params(model):,} parameters, output {tuple(out.shape)}")

    model = build("small").eval()
    npz = Path("data/encoded/val.npz")
    if npz.exists():
        split = EncodedSplit(str(npz), limit=2000)
        x, y = next(split.batches(256, shuffle=False))
        with torch.no_grad():
            loss = F.cross_entropy(model(x), y).item()
        print(f"ok  untrained loss {loss:.2f}  (random guessing = ln({NUM_MOVES}) = {math.log(NUM_MOVES):.2f})")
        assert 7.5 < loss < 12.0


def test_overfit_one_batch(n=64, steps=300):
    """If the network + labels + backprop are wired correctly, it must memorise 64 positions."""
    npz = Path("data/encoded/val.npz")
    if not npz.exists():
        print("skip overfit test (no data/encoded/val.npz)")
        return
    torch.manual_seed(0)
    split = EncodedSplit(str(npz), limit=20000)
    # use distinct positions only (the start position appears with different labels)
    _, first = np.unique(split.bb, axis=0, return_index=True)
    idx = np.sort(first)[:n]
    x, y = split.get(idx)
    model = build("small")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.time()
    for step in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == steps - 1:
            print(f"    step {step:3d}  loss {loss.item():.4f}")
    with torch.no_grad():       # same BatchNorm mode as training: we are memorising this one batch
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"ok  overfit {n} positions: final loss {loss.item():.4f}, "
          f"accuracy {acc:.0%}, {time.time() - t0:.0f}s")
    assert loss.item() < 0.2 and acc > 0.9


if __name__ == "__main__":
    test_special_positions()
    test_mapping_fuzz()
    test_real_labels()
    test_model_shapes_and_init_loss()
    test_overfit_one_batch()
    print("\nALL TESTS PASSED")