import pandas as pd
from src.validation.validate import validate_data
from src.features.build_features import build_features,build_inference_features
from src.evaluation.evaluate import evaluate_ranking


def test_validation_quarantines_bad_rating_without_double_classifying(tmp_path):
    users=pd.DataFrame({'user_id':[1],'age':[30],'region':['north']})
    items=pd.DataFrame({'movie_id':[1],'title':['A'],'genre':['Drama'],'release_year':[2020]})
    interactions=pd.DataFrame({'user_id':[1,1],'movie_id':[1,1],'rating':[None,5],'timestamp':['2024-01-01','2024-01-02']})
    clean,errors,report=validate_data(users,items,interactions,tmp_path)
    assert len(clean)==1
    assert report['invalid_interactions']==1
    assert report['invalid_ratings']==0
    assert 'missing_or_non_numeric_required_value' in set(errors.error_reason)


def test_training_and_inference_features_share_definitions(tmp_path):
    users=pd.DataFrame({'user_id':[1],'age':[30],'region':['north']})
    items=pd.DataFrame({'movie_id':[1,2],'title':['A','B'],'genre':['Drama','Comedy'],'release_year':[2020,2021]})
    hist=pd.DataFrame({'user_id':[1,1],'movie_id':[1,2],'rating':[5.,4.],'timestamp':pd.to_datetime(['2024-01-01','2024-01-02'],utc=True)})
    user,item,interaction,catalog=build_features(hist,users,items,tmp_path)
    inf_user,inf_item,ref=build_inference_features([1],[2],hist,users,items)
    assert 'user_preferred_genre' in user.columns
    assert 'user_days_since_last_interaction' in user.columns
    assert inf_user.iloc[0].user_activity_count==2
    assert inf_item.iloc[0].movie_id==2
    assert 'user_preferred_genre' in set(catalog.feature_name)


def test_ndcg_uses_independent_ideal_ranking():
    class Model:
        def recommend(self,user,seen,all_ids,k=5): return [2,3]
    train=pd.DataFrame({'user_id':[1],'movie_id':[1],'rating':[3.],'timestamp':pd.to_datetime(['2024-01-01'],utc=True)})
    test=pd.DataFrame({'user_id':[1,1],'movie_id':[2,4],'rating':[5.,5.],'timestamp':pd.to_datetime(['2024-01-02','2024-01-03'],utc=True)})
    items=pd.DataFrame({'movie_id':[1,2,3,4],'genre':['A','A','B','B']})
    m=evaluate_ranking(Model(),train,test,[1,2,3,4],items,k=2)
    assert 0 < m['ndcg_at_k'] < 1
    assert m['coverage']==0.5


def test_feature_aware_svd_uses_training_features():
    from src.training.train_models import SVDRecommender
    train = pd.DataFrame({
        'user_id': [1, 1, 2, 2], 'movie_id': [10, 11, 10, 12],
        'rating': [5., 4., 2., 5.],
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'], utc=True)
    })
    user_features = pd.DataFrame({'user_id': [1, 2], 'user_preferred_genre': ['Comedy', 'Drama'], 'user_activity_count': [2, 2]})
    item_features = pd.DataFrame({
        'movie_id': [10, 11, 12, 13], 'genre': ['Drama', 'Comedy', 'Drama', 'Comedy'],
        'item_popularity': [2, 1, 1, 0]
    })
    model = SVDRecommender(n_components=1).fit(train, user_features, item_features)
    result = model.recommend(1, {10, 11}, [10, 11, 12, 13], k=2)
    assert len(result) == 2
    assert set(result).issubset({12, 13})
    assert model.user_feature_map[1]['user_preferred_genre'] == 'Comedy'


def test_end_to_end_recommendation_example_runs_after_pipeline():
    from pipeline_tasks import TASKS
    from recommend import recommend_for_user
    for stage in ['ingest', 'validate', 'prepare', 'transform', 'features', 'train']:
        TASKS[stage]()
    result = recommend_for_user(user_id=1, k=5, model_name='svd')
    assert len(result) <= 5
    assert list(result.columns) == ['rank', 'movie_id', 'title', 'genre', 'release_year', 'model', 'feature_reference_time']
    assert result['rank'].is_monotonic_increasing


def test_invalid_metadata_is_quarantined_and_removed(tmp_path):
    users = pd.DataFrame({
        'user_id': [1, 2, 3], 'age': [30, 0, 'not-an-age'],
        'region': ['north', 'south', 'east']
    })
    items = pd.DataFrame({
        'movie_id': [10, 11], 'title': ['A', 'B'], 'genre': ['Drama', 'Comedy'],
        'release_year': [2020, 1700]
    })
    interactions = pd.DataFrame({
        'user_id': [1], 'movie_id': [10], 'rating': [5],
        'timestamp': ['2024-01-01T00:00:00Z']
    })
    clean, errors, report = validate_data(users, items, interactions, tmp_path)
    assert users.iloc[0].user_id in set(clean.user_id)
    assert report['users_valid'] == 1
    assert report['items_valid'] == 1
    assert report['users_invalid'] == 2
    assert report['items_invalid'] == 1
    assert {'invalid_age_type_or_range', 'invalid_release_year_type_or_range'} <= set(errors.error_reason)


def test_interaction_duplicates_use_normalized_utc_timestamps(tmp_path):
    users = pd.DataFrame({'user_id': [1], 'age': [30], 'region': ['north']})
    items = pd.DataFrame({'movie_id': [10], 'title': ['A'], 'genre': ['Drama'], 'release_year': [2020]})
    interactions = pd.DataFrame({
        'user_id': ['1', 1], 'movie_id': [10, '10'], 'rating': ['5', 4],
        'timestamp': ['2024-01-01T00:00:00Z', '2023-12-31T19:00:00-05:00']
    })
    clean, errors, report = validate_data(users, items, interactions, tmp_path)
    assert len(clean) == 1
    assert report['duplicate_interactions'] == 1
    assert 'duplicate_interaction' in set(errors.error_reason)


def test_experiment_tracker_records_unique_run_and_artifact_checksums(tmp_path):
    from src.tracking import log_experiment
    metrics = pd.DataFrame([{'model': 'Popularity', 'precision_at_k': 0.2}])
    models = tmp_path / 'models'
    reports = tmp_path / 'reports'
    models.mkdir(); reports.mkdir()
    (models / 'popularity_model.joblib').write_bytes(b'model-binary')
    (reports / 'model_comparison.csv').write_text('model,precision_at_k\nPopularity,0.2\n')
    result = log_experiment(metrics, {'k': 5}, reports, models)
    assert result['run_id']
    assert result['metrics'][0]['precision_at_k'] == 0.2
    assert any(a['name'] == 'popularity_model.joblib' and a['sha256'] for a in result['artifacts'])
