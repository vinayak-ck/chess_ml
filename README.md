# Chess policy network: training guide (Sprint 4)

Trains a residual CNN that predicts the move a strong human would play. The whole dataset is
kept on the GPU (about 700 MB), batches are built on the GPU, and training uses fp16 mixed
precision, so a 6 GB card is plenty.

## 1. One-time setup (Windows PowerShell)

```powershell
git clone <REPO-URL>
cd <REPO-FOLDER>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# PyTorch with CUDA. Plain `pip install torch` is CPU-ONLY on Windows, so use this instead:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# check that the GPU is visible (must print True and the GPU name)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- RTX 50-series cards need a newer CUDA build than cu126: pick the right command at
  https://pytorch.org/get-started/locally/ (cu128 or cu130).
- Use Python 3.10 to 3.12 and a reasonably recent NVIDIA driver.

## 2. Get the data (into `data/encoded/`)

**Option A: download (fast).** The owner uploads `train.npz`, `val.npz`, `test.npz` once
(see "Uploading the dataset" below) and sets the URL in `src/download_data.py`:

```powershell
python src/download_data.py
# or without editing the file:
python src/download_data.py --base-url https://github.com/<user>/<repo>/releases/download/data-v1
```

**Option B: rebuild from the public Lichess dump (about 1 hour 10 min, same result).**

```powershell
python src/extract_positions.py "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst" --max-games 100000
python src/encode_dataset.py
```

### Uploading the dataset (owner, once)
GitHub repo -> Releases -> Draft a new release -> tag `data-v1` -> drag in the three `.npz`
files from `data/encoded/` -> Publish. Each file may be up to 2 GB. The base URL is then
`https://github.com/<user>/<repo>/releases/download/data-v1`. (The repo has to be public for
anonymous downloads; otherwise use a public Hugging Face dataset repo and the base URL
`https://huggingface.co/datasets/<user>/<name>/resolve/main`.)

## 3. Check everything before the long run

```powershell
python src/test_gpu_data.py
python src/train.py --preset small --epochs 1 --train-limit 50000 --val-limit 10000 --name smoke
```

The first must print `ALL TESTS PASSED` with `device: cuda`. The second takes a minute or two;
look at the `pos_s` number in the progress bar (positions per second) and the "peak GPU memory"
line at the end. Then delete the test run: `rmdir /s runs\smoke` and `del models\smoke.pt`.

## 4. Train for real

```powershell
python src/train.py --preset base --epochs 8 --name base_v1
```

- Keep the laptop plugged in, set the Windows power mode to best performance, and close
  anything else using the GPU. `nvidia-smi` in a second window shows usage.
- The progress bar shows loss, learning rate, `pos_s` and the estimated minutes left.
- Every 2,000 steps: a quick validation check. Every epoch: full validation, and
  `runs/base_v1/best_model.pt` is updated whenever validation top-1 improves.
- **Stop any time** with Ctrl+C. Progress is saved. Continue with the same command plus `--resume`:
  `python src/train.py --preset base --epochs 8 --name base_v1 --resume`
- Limited time? `--time-limit-min 90` stops cleanly after 90 minutes (resume later the same way).
  Always keep `--epochs` the same when resuming, because the learning-rate schedule depends on it.

| Problem | Fix |
|---|---|
| CUDA out of memory | `--batch-size 512` |
| loss becomes NaN | `--amp off` (and/or `--lr 1e-3`) |
| `cuda.is_available()` is False | you installed the CPU torch: redo step 1 |
| "last.pt already exists" | add `--resume`, or choose a new `--name` |

## 5. Send the results back

```powershell
python src/plot_metrics.py --name base_v1
git add models/base_v1.pt
git commit -m "trained policy network base_v1"
git push
```

`models/base_v1.pt` is about 41 MB, under GitHub's 100 MB file limit. Also send
`runs/base_v1/metrics.csv`, `runs/base_v1/final_report.json` and `runs/base_v1/curves.png`.
(`runs/` is git-ignored because `last.pt` is about 125 MB.)

## Files

| File | Purpose |
|---|---|
| `src/train.py` | the training loop (AMP, resume, eval, export) |
| `src/gpu_data.py` | puts a split on the GPU and unpacks bitboards into planes there |
| `src/checkpoint.py` | save / load helpers |
| `src/test_gpu_data.py` | proves the GPU unpacker equals the reference encoder |
| `src/download_data.py` | fetches `data/encoded/*.npz` |
| `src/plot_metrics.py` | loss and accuracy curves |
