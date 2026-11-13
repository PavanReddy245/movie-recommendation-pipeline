"""Cleaning and chronological train/test split."""
from pathlib import Path
import pandas as pd


def prepare_data(clean_interactions, processed_dir: str, train_fraction: float = 0.8):
    """Sort events chronologically and split without using future events for training."""
    if not 0 < float(train_fraction) < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(clean_interactions.columns)
    if missing:
        raise ValueError(f"Clean interactions missing columns: {sorted(missing)}")
    if len(clean_interactions) < 2:
        raise ValueError("At least two valid interactions are required for a train/test split")

    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = clean_interactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise", utc=True)
    data = data.sort_values(["timestamp", "user_id", "movie_id"], kind="mergesort").reset_index(drop=True)
    split_idx = max(1, min(len(data) - 1, int(len(data) * float(train_fraction))))
    train, test = data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()
    if train.timestamp.max() > test.timestamp.min():
        raise AssertionError("Chronological split invariant violated")
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    data.to_csv(out / "clean_interactions.csv", index=False)
    return train, test
