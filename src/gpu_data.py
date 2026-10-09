"""Sprint 4: keep a whole encoded split on the GPU and build batches there.

Why: the compact format is only ~100 bytes/position, so the full train split (~600 MB)
fits easily in a 6 GB card. Batches are then sliced and expanded to planes on the GPU,
so training never waits on the CPU or on a DataLoader.
"""
import numpy as np
import torch
import torch.nn.functional as F

from encode import NUM_PLANES
from moves import move_to_index


def unpack_gpu(bb, castle, ep, wtm):
    """Torch twin of encode.unpack. Works on any device.
    bb (B,12) int64 | castle (B,) | ep (B,) | wtm (B,)  ->  float32 (B, 18, 8, 8)
    """
    B = bb.shape[0]
    shifts = torch.arange(64, device=bb.device, dtype=torch.int64)
    # bb holds the raw bits of uint64 bitboards. For int64, (x >> k) & 1 is still bit k
    # (arithmetic shift only changes the high bits, which the & 1 throws away).
    bits = (bb.unsqueeze(-1) >> shifts) & 1                       # (B, 12, 64)

    x = torch.zeros(B, NUM_PLANES, 64, device=bb.device, dtype=torch.float32)
    x[:, :12] = bits.float()
    castle = castle.long()
    for k in range(4):
        x[:, 12 + k] = ((castle >> k) & 1).float().unsqueeze(1)
    ep = ep.long()
    has_ep = (ep >= 0).float().unsqueeze(1)
    x[:, 16] = F.one_hot(ep.clamp(min=0), 64).float() * has_ep    # no GPU sync needed
    x[:, 17] = wtm.float().unsqueeze(1)
    return x.view(B, NUM_PLANES, 8, 8)


class GpuSplit:
    def __init__(self, path, device, limit=None):
        d = np.load(path)
        sl = slice(0, limit)
        bb = np.ascontiguousarray(d["bb"][sl])
        self.bb = torch.from_numpy(bb.view(np.int64)).to(device)           # same bits, signed view
        self.castle = torch.from_numpy(np.ascontiguousarray(d["castle"][sl])).to(device)   # uint8
        self.ep = torch.from_numpy(np.ascontiguousarray(d["ep"][sl])).to(device)           # int8
        self.wtm = torch.from_numpy(np.ascontiguousarray(d["wtm"][sl])).to(device)         # uint8
        y = move_to_index(d["mv"][sl]).astype(np.int16)                    # classes < 4168 fit in int16
        self.y = torch.from_numpy(y).to(device)
        self.device = device

    def __len__(self):
        return self.y.numel()

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.bb, self.castle, self.ep, self.wtm, self.y))

    def get(self, idx):
        """idx: LongTensor of row numbers on the same device -> (x float32, y int64)."""
        x = unpack_gpu(self.bb[idx], self.castle[idx], self.ep[idx], self.wtm[idx])
        return x, self.y[idx].long()
