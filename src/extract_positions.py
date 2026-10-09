"""Sprint 1: stream a Lichess PGN dump and extract (FEN, move) rows."""
import argparse
import csv
import gzip
import io
from pathlib import Path

import chess.pgn
import requests
import zstandard as zstd
from tqdm import tqdm


def open_stream(source: str):
    """Text stream of PGN from a URL or a local .pgn / .pgn.zst file."""
    if source.startswith("http"):
        resp = requests.get(source, stream=True, timeout=60)
        resp.raise_for_status()
        raw = resp.raw
    else:
        raw = open(source, "rb")

    if source.endswith(".zst"):
        raw = zstd.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8")


def game_passes(game, min_elo: int, min_base_seconds: int) -> bool:
    h = game.headers
    if game.errors:                      # illegal/garbled moves in the PGN
        return False
    try:
        white, black = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return False
    if min(white, black) < min_elo:
        return False
    if h.get("Termination") != "Normal":
        return False
    tc = h.get("TimeControl", "-")       # e.g. "300+3"
    if "+" not in tc:
        return False
    if int(tc.split("+")[0]) < min_base_seconds:
        return False
    return True


def split_for(game_index: int) -> str:
    r = game_index % 10                  # deterministic 80/10/10 by game
    return "val" if r == 0 else "test" if r == 1 else "train"


def extract(game, game_id: int):
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        yield (game_id, ply, board.fen(), move.uci())   # position BEFORE the move
        board.push(move)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="URL or local path to a .pgn / .pgn.zst")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--max-games", type=int, default=100_000)
    p.add_argument("--min-elo", type=int, default=1800)
    p.add_argument("--min-base-seconds", type=int, default=180)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: gzip.open(out / f"{s}.csv.gz", "wt", newline="")
             for s in ("train", "val", "test")}
    writers = {s: csv.writer(f) for s, f in files.items()}
    for w in writers.values():
        w.writerow(["game_id", "ply", "fen", "move"])

    rows = {"train": 0, "val": 0, "test": 0}
    seen = accepted = 0
    stream = open_stream(args.source)
    pbar = tqdm(total=args.max_games, desc="accepted games")
    try:
        while accepted < args.max_games:
            game = chess.pgn.read_game(stream)
            if game is None:             # end of file
                break
            seen += 1
            if not game_passes(game, args.min_elo, args.min_base_seconds):
                continue
            split = split_for(accepted)
            for row in extract(game, accepted):
                writers[split].writerow(row)
                rows[split] += 1
            accepted += 1
            pbar.update(1)
    finally:
        pbar.close()
        for f in files.values():
            f.close()

    print(f"read {seen} games, kept {accepted}")
    print("rows per split:", rows)


if __name__ == "__main__":
    main()