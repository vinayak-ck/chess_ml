"""Sprint 3: load the encoded .npz files and serve (x, y) batches."""
import numpy as np
import torch

from encode import unpack
from moves import move_to_index


class EncodedSplit:
    """One split (train / val / test) held in memory in compact form (~100 bytes/row)."""

    def __init__(self, path: str, limit: int | None = None):
        d = np.load(path)
        sl = slice(0, limit)
        self.bb = d["bb"][sl]
        self.castle = d["castle"][sl]
        self.ep = d["ep"][sl]
        self.wtm = d["wtm"][sl]
        self.y = move_to_index(d["mv"][sl])          # (N,) int64 class labels

    def __len__(self) -> int:
        return len(self.y)

    def get(self, idx: np.ndarray):
        """Rows `idx` -> (x float32 (B,18,8,8), y int64 (B,)) as torch tensors."""
        x = unpack(self.bb[idx], self.castle[idx], self.ep[idx], self.wtm[idx])
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0,
                drop_last: bool = True):
        n = len(self)
        order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
        stop = n - batch_size + 1 if drop_last else n
        for i in range(0, stop, batch_size):
            yield self.get(order[i:i + batch_size])