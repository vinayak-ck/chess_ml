import chess
import pandas as pd

df = pd.read_csv("data/raw/train.csv.gz")
print("shape:", df.shape)
print(df.head(), "\n")

# 1. every label must be a legal move in its own position
sample = df.sample(min(2000, len(df)), random_state=0)
illegal = sum(
    chess.Move.from_uci(m) not in chess.Board(f).legal_moves
    for f, m in zip(sample.fen, sample.move)
)
print("illegal moves in sample:", illegal, "(must be 0)")

# 2. White and Black to move should be about 50/50
print(df.fen.str.split().str[1].value_counts(normalize=True), "\n")

# 3. most common labels (expect e2e4, d2d4, g1f3 ... near the top)
print(df.move.value_counts().head(10), "\n")

# 4. how many distinct moves does the model need to choose between?
print("distinct move labels:", df.move.nunique())

# 5. splits must not share games
val = pd.read_csv("data/raw/val.csv.gz")
test = pd.read_csv("data/raw/test.csv.gz")
ids = [set(d.game_id) for d in (df, val, test)]
print("game overlap:", len(ids[0] & ids[1]), len(ids[0] & ids[2]), len(ids[1] & ids[2]), "(all must be 0)")