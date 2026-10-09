"""Sprint 4: train the policy network. GPU-friendly and resumable.

Examples
  python src/train.py --preset small --epochs 1 --train-limit 50000 --val-limit 10000 --name smoke
  python src/train.py --preset base  --epochs 8 --name base_v1
  python src/train.py --preset base  --epochs 8 --name base_v1 --resume     # continue after a stop
"""
import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from checkpoint import load_training_state, save_training_state, save_weights
from gpu_data import GpuSplit
from model import PRESETS, PolicyNet, count_params


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="base_v1", help="run name -> runs/<name>/ and models/<name>.pt")
    p.add_argument("--preset", default="base", choices=sorted(PRESETS))
    p.add_argument("--data-dir", default="data/encoded")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024, help="lower to 512 if you hit CUDA out-of-memory")
    p.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="fp16 mixed precision")
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--eval-every", type=int, default=2000, help="steps between quick validation checks")
    p.add_argument("--quick-val", type=int, default=50000, help="positions used in a quick validation")
    p.add_argument("--ckpt-every", type=int, default=3000, help="steps between resumable checkpoints")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--train-limit", type=int, default=None, help="use only the first N training rows")
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--time-limit-min", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def lr_at(step, total, peak, warmup, floor=0.02):
    """Linear warm-up, then cosine decay down to floor*peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def resolve_amp(mode, device):
    if device != "cuda" or mode == "off":
        return False
    if mode == "on":
        return True
    name = torch.cuda.get_device_name(0)
    return torch.cuda.get_device_capability(0) >= (7, 0) and "GTX 16" not in name


@torch.no_grad()
def evaluate(model, split, device, use_amp, channels_last, idx=None, batch=4096):
    """Loss + top-1/3/5 accuracy over a split (or a subset given by idx)."""
    model.eval()
    n = len(split) if idx is None else len(idx)
    total, c1, c3, c5 = 0.0, 0, 0, 0
    for i in range(0, n, batch):
        ids = torch.arange(i, min(i + batch, n), device=device) if idx is None else idx[i:i + batch]
        x, y = split.get(ids)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        total += F.cross_entropy(logits, y, reduction="sum").item()
        hit = logits.topk(5, dim=1).indices == y.unsqueeze(1)
        c1 += hit[:, :1].any(1).sum().item()
        c3 += hit[:, :3].any(1).sum().item()
        c5 += hit.any(1).sum().item()
    model.train()
    return {"loss": total / n, "top1": c1 / n, "top3": c3 / n, "top5": c5 / n}


class MetricsLog:
    FIELDS = ["step", "epoch", "kind", "loss", "top1", "top3", "top5", "lr", "minutes"]

    def __init__(self, path):
        new = not path.exists()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def add(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.f.flush()


def fmt(m):
    return f"loss {m['loss']:.3f} | top1 {m['top1']:.1%} | top3 {m['top3']:.1%} | top5 {m['top5']:.1%}"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA GPU found, training on CPU (very slow for --preset base).")
        print("         On Windows, plain `pip install torch` is CPU-only. See README.md for the CUDA install.")
    use_amp = resolve_amp(args.amp, device)
    channels_last = use_amp and not args.no_channels_last

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best_model.pt"
    if last_path.exists() and not args.resume:
        raise SystemExit(f"{last_path} already exists. Use --resume to continue it, or pick a new --name.")

    config = {"head_channels": 32, **PRESETS[args.preset]}
    model = PolicyNet(**config).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    decay = [p for p in model.parameters() if p.ndim >= 2]       # conv + linear weights
    no_decay = [p for p in model.parameters() if p.ndim < 2]     # BatchNorm params and biases
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    print("loading data onto the device ...")
    train = GpuSplit(Path(args.data_dir) / "train.npz", device, args.train_limit)
    val = GpuSplit(Path(args.data_dir) / "val.npz", device, args.val_limit)
    n = len(train)
    steps_per_epoch = n // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit("--batch-size is larger than the training set; lower it or raise --train-limit.")
    total_steps = steps_per_epoch * args.epochs
    gq = torch.Generator()
    gq.manual_seed(1234)
    quick_idx = None
    if len(val) > args.quick_val:
        quick_idx = torch.randperm(len(val), generator=gq)[:args.quick_val].to(device)

    step, best = 0, {"top1": -1.0}
    if args.resume and last_path.exists():
        ck = load_training_state(last_path, device)
        if ck["config"] != config:
            raise SystemExit("checkpoint was trained with a different --preset; cannot resume it.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from step {step}")

    (run_dir / "config.json").write_text(json.dumps({**vars(args), "model": config}, indent=2))
    print(f"model: {args.preset} {config} -> {count_params(model):,} parameters")
    print(f"train {n:,} positions | val {len(val):,} | batch {args.batch_size} | "
          f"{steps_per_epoch:,} steps/epoch | {total_steps:,} steps total")
    print(f"mixed precision (fp16): {use_amp} | channels_last: {channels_last} | "
          f"data on device: {(train.nbytes() + val.nbytes()) / 1e6:.0f} MB")

    log = MetricsLog(run_dir / "metrics.csv")
    t0 = time.time()
    minutes = lambda: (time.time() - t0) / 60
    start_step = step
    start_epoch = step // steps_per_epoch if steps_per_epoch else 0
    loss_acc, acc_n = torch.zeros((), device=device), 0
    stop_reason = None
    pbar = tqdm(total=total_steps, initial=step, unit="step", dynamic_ncols=True)

    def save_last():
        save_training_state(last_path, model, opt, scaler, step, best, config, vars(args))

    def do_eval(kind, idx=None):
        m = evaluate(model, val, device, use_amp, channels_last, idx=idx)
        log.add(step=step, epoch=round(step / steps_per_epoch, 3), kind=kind, minutes=round(minutes(), 2),
                lr=opt.param_groups[0]["lr"], **m)
        pbar.write(f"[step {step:>7,} | epoch {step / steps_per_epoch:5.2f}] {kind:9s} {fmt(m)}")
        return m

    model.train()
    try:
        for epoch in range(start_epoch, args.epochs):
            g = torch.Generator()
            g.manual_seed(args.seed + epoch)
            perm = torch.randperm(n, generator=g).to(device)
            for b in range(step - epoch * steps_per_epoch, steps_per_epoch):
                lr = lr_at(step, total_steps, args.lr, args.warmup_steps)
                for group in opt.param_groups:
                    group["lr"] = lr

                x, y = train.get(perm[b * args.batch_size:(b + 1) * args.batch_size])
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()

                step += 1
                pbar.update(1)
                loss_acc += loss.detach()
                acc_n += 1

                if step % args.log_every == 0:
                    avg = (loss_acc / acc_n).item()
                    loss_acc, acc_n = torch.zeros((), device=device), 0
                    if not math.isfinite(avg):
                        raise SystemExit("loss became NaN/inf. Try --amp off or a lower --lr (e.g. 1e-3).")
                    log.add(step=step, epoch=round(step / steps_per_epoch, 3), kind="train",
                            loss=avg, lr=lr, minutes=round(minutes(), 2))
                    rate = (step - start_step) * args.batch_size / max(1e-9, time.time() - t0)
                    eta = (total_steps - step) * args.batch_size / max(1e-9, rate) / 60
                    pbar.set_postfix(loss=f"{avg:.3f}", lr=f"{lr:.1e}", pos_s=f"{rate:,.0f}", eta_min=f"{eta:.0f}")

                if step % args.eval_every == 0:
                    do_eval("val_quick", quick_idx)
                if step % args.ckpt_every == 0:
                    save_last()
                if args.time_limit_min and minutes() >= args.time_limit_min:
                    stop_reason = f"time limit of {args.time_limit_min} min reached"
                    break

            if stop_reason:
                break
            m = do_eval("val_full")                                  # end of epoch
            if m["top1"] > best["top1"]:
                best = {**m, "step": step}
                save_weights(best_path, model, config, best)
                pbar.write(f"  new best model saved (val top1 {m['top1']:.1%})")
            save_last()
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"

    pbar.close()
    save_last()
    if device == "cuda":
        print(f"peak GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    if stop_reason:
        print(f"\nStopped: {stop_reason}. Progress saved at step {step:,}.")
        print(f"Continue with the SAME command you started with, plus --resume, e.g.:\n"
              f"  python src/train.py --name {args.name} --preset {args.preset} "
              f"--epochs {args.epochs} --batch-size {args.batch_size} --resume")
        return

    # ---- finished all epochs: final test-set score for the best model, then export ----
    report = {"best_val": best, "parameters": count_params(model), "minutes_this_session": round(minutes(), 1)}
    test_path = Path(args.data_dir) / "test.npz"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        if test_path.exists():
            test = GpuSplit(test_path, device)
            report["test"] = evaluate(model, test, device, use_amp, channels_last)
            print(f"\nTEST SET (best model): {fmt(report['test'])}")
        out = Path("models") / f"{args.name}.pt"
        out.parent.mkdir(exist_ok=True)
        shutil.copy(best_path, out)
        print(f"exported weights -> {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    (run_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    print("done. Plot the curves with:  python src/plot_metrics.py --name", args.name)


if __name__ == "__main__":
    main()
