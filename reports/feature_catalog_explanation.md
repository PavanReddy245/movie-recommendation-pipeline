# Feature Catalog

The feature catalog is maintained in `data/features/feature_catalog.csv`. All interaction aggregates are calculated from the training/history data available at the feature reference timestamp to avoid future-data leakage.

| Entity | Feature | Meaning |
|---|---|---|
| User | user_avg_rating | Mean historical rating |
| User | user_activity_count | Number of historical interactions |
| User | user_last_interaction | Latest historical interaction time |
| User | user_days_since_last_interaction | Recency at reference time |
| User | user_preferred_genre | Most frequently interacted genre |
| User | age, region | User metadata |
| Item | item_popularity | Historical interaction count |
| Item | item_avg_rating | Mean historical rating |
| Item | genre, title, release_year | Item metadata |
| Interaction | rating, timestamp | Event-level values |
| Interaction | days_since_reference | Event recency relative to reference time |

`build_features()` creates training feature tables. `build_inference_features()` uses the same aggregation definitions for requested user/item IDs and a caller-supplied history snapshot. For production use, persist the feature reference timestamp and ensure training and serving use identical point-in-time data semantics.
