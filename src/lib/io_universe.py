from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

NODE_MAP_PATH = Path("artifacts/snapshots/node_map.json")

def load_node_map(path: Path = NODE_MAP_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def load_weekly_returns_aligned(
    returns_path: str = "data/processed/weekly_returns.parquet",
    node_map_path: Path = NODE_MAP_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(returns_path).sort_index()
    node_map = load_node_map(node_map_path)
    tickers = node_map["tickers_in_order"]

    # Fuerza orden y presencia exacta
    df = df[tickers]

    return df




