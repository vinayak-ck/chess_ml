"""Sprint 4: download the pre-encoded dataset (data/encoded/*.npz) from a URL.

The three files (train.npz ~580 MB, val.npz ~72 MB, test.npz ~72 MB) are too big for git.
Upload them once (GitHub Release or Hugging Face dataset, see README.md), put the base URL
below, and any machine can fetch them with:   python src/download_data.py
"""
import argparse
from pathlib import Path

import requests
from tqdm import tqdm

# >>> EDIT THIS after you upload the files. Examples:
#   GitHub release : https://github.com/<user>/<repo>/releases/download/data-v1
#   Hugging Face   : https://huggingface.co/datasets/<user>/<name>/resolve/main
DEFAULT_BASE_URL = "https://github.com/vinayak-ck/chess_ml/releases/tag/data-v1"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--out", default="data/encoded")
    p.add_argument("--files", nargs="+", default=["train.npz", "val.npz", "test.npz"])
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = p.parse_args()

    if not args.base_url:
        raise SystemExit("No base URL set. Pass --base-url <url> or edit DEFAULT_BASE_URL in this file.\n"
                         "No upload yet? Rebuild the data instead (see README.md, 'Option B').")
    for name in args.files:
        dest = Path(args.out) / name
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            print(f"{name}: already there, skipping")
            continue
        download(f"{args.base_url.rstrip('/')}/{name}", dest)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
