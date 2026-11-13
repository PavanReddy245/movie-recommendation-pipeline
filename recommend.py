"""Generate top-K movie recommendations from saved models and inference-time features."""
import argparse
from pathlib import Path
import pandas as pd
from joblib import load
from src.features.build_features import build_inference_features

ROOT = Path(__file__).resolve().parent

def recommend_for_user(user_id: int, k: int = 5, model_name: str = 'svd'):
    processed = ROOT / 'data' / 'processed'
    users = pd.read_csv(processed / 'validated_users.csv')
    items = pd.read_csv(processed / 'validated_items.csv')
    history = pd.read_csv(processed / 'train.csv', parse_dates=['timestamp'])
    model_file = 'svd_model.joblib' if model_name == 'svd' else 'popularity_model.joblib'
    model = load(ROOT / 'models' / model_file)
    request_item_ids = items.movie_id.tolist()
    user_features, item_features, reference_time = build_inference_features(
        [user_id], request_item_ids, history, users, items
    )
    if model_name == 'popularity':
        recs = model.recommend(user_id, set(history.loc[history.user_id == user_id, 'movie_id']), request_item_ids, k=k)
    else:
        seen = set(history.loc[history.user_id == user_id, 'movie_id'])
        recs = model.recommend(user_id, seen, request_item_ids, k=k,
                               user_features=user_features, item_features=item_features)
    output = items[items.movie_id.isin(recs)].copy()
    output['rank'] = output.movie_id.map({movie_id: rank for rank, movie_id in enumerate(recs, 1)})
    output = output.sort_values('rank')
    output['model'] = model_name
    output['feature_reference_time'] = reference_time.isoformat()
    return output[['rank', 'movie_id', 'title', 'genre', 'release_year', 'model', 'feature_reference_time']]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--model', choices=['svd', 'popularity'], default='svd')
    parser.add_argument('--output', default='reports/sample_recommendations.csv')
    args = parser.parse_args()
    result = recommend_for_user(args.user_id, args.k, args.model)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f'\nSaved: {path}')
