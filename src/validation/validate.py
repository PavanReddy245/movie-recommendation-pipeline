"""Schema, domain, duplicate and referential-integrity validation."""
from pathlib import Path
import pandas as pd


def _record_issues(frame, dataset, masks, errors):
    """Append one quarantined row per record/issue and return the union mask."""
    invalid = pd.Series(False, index=frame.index, dtype=bool)
    for reason, mask in masks.items():
        mask = mask.fillna(False).astype(bool)
        invalid |= mask
        if mask.any():
            issue = frame.loc[mask].copy()
            issue["dataset"] = dataset
            issue["error_reason"] = reason
            errors.append(issue)
    return invalid


def _metadata_validation(frame, key, label, errors):
    x = frame.copy()
    schemas = {
        "users": ["user_id", "age", "region"],
        "items": ["movie_id", "title", "genre", "release_year"],
    }
    required = schemas[label]
    missing_cols = [c for c in required if c not in x.columns]
    if missing_cols:
        raise ValueError(f"{label} dataset missing required columns: {missing_cols}")

    for col in required:
        if x[col].dtype == "object":
            x[col] = x[col].replace(r"^\s*$", pd.NA, regex=True)

    numeric_key = pd.to_numeric(x[key], errors="coerce")
    masks = {
        "missing_required_metadata": x[required].isna().any(axis=1),
        "invalid_data_type_or_identifier": numeric_key.isna() | (numeric_key % 1 != 0) | (numeric_key <= 0),
        f"duplicate_{key}": numeric_key.duplicated(keep="first"),
    }

    if label == "users":
        age = pd.to_numeric(x["age"], errors="coerce")
        masks["invalid_age_type_or_range"] = age.isna() | (age % 1 != 0) | ~age.between(1, 120)
        masks["invalid_region_type"] = ~x["region"].map(lambda v: isinstance(v, str) and bool(v.strip()))
    else:
        year = pd.to_numeric(x["release_year"], errors="coerce")
        masks["invalid_release_year_type_or_range"] = year.isna() | (year % 1 != 0) | ~year.between(1888, 2100)
        for col in ("title", "genre"):
            masks[f"invalid_{col}_type"] = ~x[col].map(lambda v: isinstance(v, str) and bool(v.strip()))

    invalid = _record_issues(x, label, masks, errors)
    valid = x.loc[~invalid].copy()
    valid[key] = pd.to_numeric(valid[key]).astype("int64")
    if label == "users":
        valid["age"] = pd.to_numeric(valid["age"]).astype("int64")
    else:
        valid["release_year"] = pd.to_numeric(valid["release_year"]).astype("int64")
    return valid


def validate_data(users, items, interactions, error_dir: str):
    """Validate all source tables and quarantine invalid records by issue."""
    errors = []
    users_ok = _metadata_validation(users, "user_id", "users", errors)
    items_ok = _metadata_validation(items, "movie_id", "items", errors)

    x = interactions.copy()
    required = ["user_id", "movie_id", "rating", "timestamp"]
    missing_cols = [c for c in required if c not in x.columns]
    if missing_cols:
        raise ValueError(f"Interactions dataset missing required columns: {missing_cols}")

    original = x.copy()
    user_id = pd.to_numeric(x["user_id"], errors="coerce")
    movie_id = pd.to_numeric(x["movie_id"], errors="coerce")
    rating = pd.to_numeric(x["rating"], errors="coerce")
    timestamp = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)

    masks = {
        "missing_or_non_numeric_required_value": x[required].isna().any(axis=1),
        "invalid_user_id_type_or_value": user_id.isna() | (user_id % 1 != 0) | (user_id <= 0),
        "invalid_movie_id_type_or_value": movie_id.isna() | (movie_id % 1 != 0) | (movie_id <= 0),
        "invalid_rating_type": x["rating"].notna() & rating.isna(),
        "rating_out_of_range": rating.notna() & ~rating.between(1, 5),
        "invalid_timestamp": timestamp.isna(),
        "unknown_user_id": user_id.notna() & ~user_id.isin(users_ok.user_id),
        "unknown_movie_id": movie_id.notna() & ~movie_id.isin(items_ok.movie_id),
    }
    normalized = pd.DataFrame({"user_id": user_id, "movie_id": movie_id, "timestamp": timestamp}, index=x.index)
    masks["duplicate_interaction"] = normalized.duplicated(
        subset=["user_id", "movie_id", "timestamp"], keep="first"
    ) & user_id.notna() & movie_id.notna() & timestamp.notna()

    invalid = _record_issues(original, "interactions", masks, errors)
    clean = pd.DataFrame({
        "user_id": user_id.loc[~invalid].astype("int64"),
        "movie_id": movie_id.loc[~invalid].astype("int64"),
        "rating": rating.loc[~invalid].astype(float),
        "timestamp": timestamp.loc[~invalid],
    }).reset_index(drop=True)

    error_dir = Path(error_dir)
    error_dir.mkdir(parents=True, exist_ok=True)
    error_df = pd.concat(errors, ignore_index=True, sort=False) if errors else pd.DataFrame()
    error_df.to_csv(error_dir / "invalid_records.csv", index=False)

    report = {
        "users_input": int(len(users)), "users_valid": int(len(users_ok)),
        "users_invalid": int(len(users) - len(users_ok)),
        "items_input": int(len(items)), "items_valid": int(len(items_ok)),
        "items_invalid": int(len(items) - len(items_ok)),
        "input_interactions": int(len(x)), "valid_interactions": int(len(clean)),
        "invalid_interactions": int(invalid.sum()),
        "invalid_issue_rows": int(sum(len(issue) for issue in errors)),
        "duplicate_interactions": int(masks["duplicate_interaction"].sum()),
        "missing_required_interaction_values": int(masks["missing_or_non_numeric_required_value"].sum()),
        "invalid_interaction_types": int((masks["invalid_user_id_type_or_value"] | masks["invalid_movie_id_type_or_value"] | masks["invalid_rating_type"]).sum()),
        "invalid_timestamps": int(masks["invalid_timestamp"].sum()),
        "invalid_ratings": int(masks["rating_out_of_range"].sum()),
        "invalid_rating_types": int(masks["invalid_rating_type"].sum()),
        "unknown_users": int(masks["unknown_user_id"].sum()),
        "unknown_movies": int(masks["unknown_movie_id"].sum()),
        "metadata_issue_rows": int(sum(len(z) for z in errors if "dataset" in z and z["dataset"].iloc[0] != "interactions")),
    }
    pd.DataFrame([report]).to_csv(error_dir / "validation_report.csv", index=False)
    return clean, error_df, report
