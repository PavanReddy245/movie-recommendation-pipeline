from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

class PopularityRecommender:
    def fit(self, train):
        self.popularity = train.groupby("movie_id").size().sort_values(ascending=False)
        self.item_ids = self.popularity.index.to_list()
        return self
    def recommend(self, user_id, seen_items, all_item_ids=None, k=5):
        return [int(i) for i in self.item_ids if i not in seen_items][:k]

class SVDRecommender:
    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
    def fit(self, train, user_features=None, item_features=None):
        self.users = sorted(train.user_id.unique().tolist())
        self.items = sorted(train.movie_id.unique().tolist())
        self.user_index = {u:i for i,u in enumerate(self.users)}
        self.item_index = {m:i for i,m in enumerate(self.items)}
        matrix = np.zeros((len(self.users), len(self.items)), dtype=float)
        counts = np.zeros_like(matrix)
        for row in train.itertuples():
            ui, ii = self.user_index[row.user_id], self.item_index[row.movie_id]
            matrix[ui,ii] += float(row.rating)
            counts[ui,ii] += 1
        observed = counts > 0
        matrix[observed] /= counts[observed]
        self.user_means = np.divide(matrix.sum(axis=1), observed.sum(axis=1),
                                    out=np.full(len(self.users), 3.0),
                                    where=observed.sum(axis=1)>0)
        # Mean-center observed ratings; unobserved entries remain zero.
        centered = np.where(observed, matrix - self.user_means[:,None], 0.0)
        n_comp = max(1, min(self.n_components, min(centered.shape)-1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        user_factors = self.svd.fit_transform(centered)
        self.pred_matrix = user_factors @ self.svd.components_ + self.user_means[:,None]
        self.pred_matrix = np.clip(self.pred_matrix, 1.0, 5.0)
        self.train_matrix = matrix
        self.observed = observed
        # Persist the same engineered feature definitions used by the feature layer.
        self.user_feature_map = {}
        if user_features is not None and not user_features.empty:
            self.user_feature_map = user_features.set_index('user_id').to_dict(orient='index')
        self.item_feature_map = {}
        if item_features is not None and not item_features.empty:
            self.item_feature_map = item_features.set_index('movie_id').to_dict(orient='index')
        popularity = item_features.set_index('movie_id').item_popularity.to_dict() if item_features is not None and 'item_popularity' in item_features else {}
        max_pop = max(popularity.values(), default=1) or 1
        self.item_popularity_score = {int(k): float(v) / float(max_pop) for k, v in popularity.items()}
        return self
    def predict(self, user_id, movie_id):
        if user_id not in self.user_index or movie_id not in self.item_index:
            return float(np.mean(self.train_matrix[self.observed])) if self.observed.any() else 3.0
        return float(self.pred_matrix[self.user_index[user_id], self.item_index[movie_id]])
    def recommend(self, user_id, seen_items, all_item_ids, k=5, user_features=None, item_features=None):
        """Hybrid ranking: collaborative-filtering score + reusable genre/popularity features."""
        user_row = None
        if user_features is not None and not user_features.empty:
            rows = user_features[user_features.user_id == user_id]
            if not rows.empty: user_row = rows.iloc[0].to_dict()
        if user_row is None: user_row = getattr(self, 'user_feature_map', {}).get(user_id, {})
        preferred = str(user_row.get('user_preferred_genre', 'unknown'))
        item_map = getattr(self, 'item_feature_map', {})
        if item_features is not None and not item_features.empty:
            item_map = item_features.set_index('movie_id').to_dict(orient='index')
        candidates = []
        for movie_id in all_item_ids:
            if movie_id in seen_items: continue
            if user_id in self.user_index and movie_id in self.item_index:
                cf_score = float(self.pred_matrix[self.user_index[user_id], self.item_index[movie_id]]) / 5.0
            else:
                cf_score = 0.6
            features = item_map.get(movie_id, {})
            genre_score = 1.0 if preferred != 'unknown' and str(features.get('genre', 'unknown')) == preferred else 0.0
            popularity_score = float(getattr(self, 'item_popularity_score', {}).get(int(movie_id), 0.0))
            # All terms are normalized to [0,1]; weights sum to one.
            score = 0.80 * cf_score + 0.15 * genre_score + 0.05 * popularity_score
            candidates.append((int(movie_id), score))
        # Greedy genre-aware re-ranking (MMR-style): retain relevance while
        # discouraging a list made up of only one genre.
        selected, genre_counts = [], {}
        remaining = candidates[:]
        while remaining and len(selected) < k:
            best = max(remaining, key=lambda pair: (
                pair[1] - 0.18 * genre_counts.get(str(item_map.get(pair[0], {}).get('genre', 'unknown')), 0),
                -pair[0]
            ))
            selected.append(best[0])
            genre = str(item_map.get(best[0], {}).get('genre', 'unknown'))
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            remaining.remove(best)
        return selected

def train_models(train, model_dir: str, config: dict, user_features=None, item_features=None):
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    popularity = PopularityRecommender().fit(train)
    svd = SVDRecommender(config["svd_components"], config["svd_random_state"]).fit(train, user_features, item_features)
    joblib.dump(popularity, out / "popularity_model.joblib")
    joblib.dump(svd, out / "svd_model.joblib")
    return popularity, svd
