from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_data(output_dir: str, config: dict):
    """Create deterministic, documented synthetic users, movies and interactions."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])

    n_users = config["synthetic_users"]
    n_items = config["synthetic_items"]
    n_interactions = config["synthetic_interactions"]

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(18, 66, n_users),
        "region": rng.choice(["north", "south", "east", "west"], n_users),
    })

    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
    items = pd.DataFrame({
        "movie_id": np.arange(1, n_items + 1),
        "title": [f"Movie {i:03d}" for i in range(1, n_items + 1)],
        "genre": rng.choice(genres, n_items),
        "release_year": rng.integers(1990, 2025, n_items),
    })

    # Create user genre preferences, then generate interactions with a preference signal.
    user_pref = rng.integers(0, len(genres), n_users)
    item_genre_idx = items["genre"].map({g:i for i,g in enumerate(genres)}).to_numpy()
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for _ in range(n_interactions):
        u_idx = int(rng.integers(0, n_users))
        preferred = user_pref[u_idx]
        candidates = np.arange(n_items)
        weights = np.where(item_genre_idx == preferred, 4.0, 1.0)
        weights = weights / weights.sum()
        i_idx = int(rng.choice(candidates, p=weights))
        rating = float(np.clip(np.rint(rng.normal(3.5 + (item_genre_idx[i_idx] == preferred)*0.8, 0.9)), 1, 5))
        ts = start + pd.Timedelta(days=int(rng.integers(0, 365)), seconds=int(rng.integers(0, 86400)))
        rows.append((u_idx + 1, i_idx + 1, rating, ts.isoformat()))

    interactions = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    # Add a few deliberate quality issues to demonstrate quarantine handling.
    bad = pd.DataFrame([
        {"user_id": 1, "movie_id": 1, "rating": np.nan, "timestamp": "2024-04-01T00:00:00+00:00"},
        {"user_id": 2, "movie_id": 2, "rating": 8, "timestamp": "2024-04-02T00:00:00+00:00"},
        {"user_id": 3, "movie_id": 3, "rating": 4, "timestamp": "not-a-date"},
        {"user_id": 9999, "movie_id": 4, "rating": 4, "timestamp": "2024-04-03T00:00:00+00:00"},
        {"user_id": 4, "movie_id": 9999, "rating": 3, "timestamp": "2024-04-04T00:00:00+00:00"},
    ])
    interactions = pd.concat([interactions, bad, interactions.iloc[[0]]], ignore_index=True)

    users.to_csv(out / "users.csv", index=False)
    items.to_csv(out / "movies.csv", index=False)
    interactions.to_csv(out / "ratings.csv", index=False)
    return users, items, interactions

def load_raw_data(raw_dir: str):
    p = Path(raw_dir)
    return (
        pd.read_csv(p / "users.csv"),
        pd.read_csv(p / "movies.csv"),
        pd.read_csv(p / "ratings.csv"),
    )
