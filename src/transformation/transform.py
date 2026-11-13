from pathlib import Path
import pandas as pd

def create_derived_datasets(train, items, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    user_summary = train.groupby("user_id").agg(
        user_activity_count=("movie_id", "count"),
        user_avg_rating=("rating", "mean"),
        user_last_interaction=("timestamp", "max")
    ).reset_index()
    item_popularity = train.groupby("movie_id").agg(
        interaction_count=("user_id", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index().merge(items[["movie_id","title","genre"]], on="movie_id", how="left")
    matrix = train.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
    user_summary.to_csv(out / "user_activity_summary.csv", index=False)
    item_popularity.to_csv(out / "item_popularity.csv", index=False)
    matrix.to_csv(out / "user_item_matrix.csv")
    return user_summary, item_popularity, matrix
