"""Sprint 2: pre-encode data/raw/*.csv.gz into compact arrays in data/encoded/*.npz."""
import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

from encode import compact, orient_move

RAW = Path("data/raw")
OUT = Path("data/encoded")


def encode_split(split: str, chunksize: int = 250_000) -> None:
    parts = {k: [] for k in ("bb", "castle", "ep", "wtm", "mv")}
    pbar = tqdm(desc=split, unit="rows")
    for chunk in pd.read_csv(RAW / f"{split}.csv.gz", usecols=["fen", "move"],
                             chunksize=chunksize):
        n = len(chunk)
        bb = np.empty((n, 12), dtype=np.uint64)
        castle = np.empty(n, dtype=np.uint8)
        ep = np.empty(n, dtype=np.int8)
        wtm = np.empty(n, dtype=np.uint8)
        mv = np.empty((n, 3), dtype=np.uint8)     # from_square, to_square, promotion (0 = none)

        for i, (fen, uci) in enumerate(zip(chunk.fen, chunk.move)):
            board = chess.Board(fen)
            bb[i], castle[i], ep[i], wtm[i] = compact(board)
            m = orient_move(chess.Move.from_uci(uci), board.turn)
            mv[i] = (m.from_square, m.to_square, m.promotion or 0)

        for k, v in zip(parts, (bb, castle, ep, wtm, mv)):
            parts[k].append(v)
        pbar.update(n)
    pbar.close()

    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez(OUT / f"{split}.npz", **arrays)
    print(f"{split}: saved {len(arrays['bb']):,} rows -> {OUT / (split + '.npz')}")


if __name__ == "__main__":
    splits = sys.argv[1:] or ["val", "test", "train"]
    for s in splits:
        encode_split(s)