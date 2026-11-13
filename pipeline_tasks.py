"""File-backed task functions shared by the CLI workflow and Airflow DAG."""
import json
from pathlib import Path
import pandas as pd
from src.ingestion.ingest import generate_synthetic_data,load_raw_data
from src.validation.validate import validate_data
from src.preparation.prepare import prepare_data
from src.transformation.transform import create_derived_datasets
from src.features.build_features import build_features
from src.training.train_models import train_models
from src.evaluation.evaluate import compare_models
from src.tracking import log_experiment

ROOT=Path(__file__).resolve().parent

def config(): return json.loads((ROOT/'configs/config.json').read_text())
def ingest():
    c=config(); generate_synthetic_data(ROOT/'data/raw',c)
    return {'users':len(pd.read_csv(ROOT/'data/raw/users.csv')),'items':len(pd.read_csv(ROOT/'data/raw/movies.csv')),'interactions':len(pd.read_csv(ROOT/'data/raw/ratings.csv'))}
def validate():
    u,i,x=load_raw_data(ROOT/'data/raw'); clean,_,report=validate_data(u,i,x,ROOT/'data/errors')
    clean.to_csv(ROOT/'data/processed/validated_interactions.csv',index=False)
    u.to_csv(ROOT/'data/processed/validated_users.csv',index=False); i.to_csv(ROOT/'data/processed/validated_items.csv',index=False)
    return report
def prepare():
    x=pd.read_csv(ROOT/'data/processed/validated_interactions.csv',parse_dates=['timestamp'])
    train,test=prepare_data(x,ROOT/'data/processed',config()['train_fraction'])
    return {'train':len(train),'test':len(test)}
def transform():
    train=pd.read_csv(ROOT/'data/processed/train.csv',parse_dates=['timestamp']); items=pd.read_csv(ROOT/'data/processed/validated_items.csv')
    create_derived_datasets(train,items,ROOT/'data/processed'); return {'status':'complete'}
def features():
    train=pd.read_csv(ROOT/'data/processed/train.csv',parse_dates=['timestamp']); users=pd.read_csv(ROOT/'data/processed/validated_users.csv'); items=pd.read_csv(ROOT/'data/processed/validated_items.csv')
    build_features(train,users,items,ROOT/'data/features'); return {'status':'complete'}
def train():
    train_df=pd.read_csv(ROOT/'data/processed/train.csv',parse_dates=['timestamp'])
    user_features=pd.read_csv(ROOT/'data/features/user_features.csv')
    item_features=pd.read_csv(ROOT/'data/features/item_features.csv')
    p,s=train_models(train_df,ROOT/'models',config(),user_features,item_features)
    return {'models':['popularity','feature_aware_personalized_svd'],'training_feature_rows':{'users':len(user_features),'items':len(item_features)}}
def evaluate():
    from joblib import load
    p=load(ROOT/'models/popularity_model.joblib'); s=load(ROOT/'models/svd_model.joblib')
    tr=pd.read_csv(ROOT/'data/processed/train.csv',parse_dates=['timestamp']); te=pd.read_csv(ROOT/'data/processed/test.csv',parse_dates=['timestamp']); it=pd.read_csv(ROOT/'data/processed/validated_items.csv')
    result=compare_models(p,s,tr,te,it,ROOT/'reports/model_comparison.csv',config())
    return result.to_dict(orient='records')
def track():
    metrics=pd.read_csv(ROOT/'reports/model_comparison.csv')
    return log_experiment(metrics,config(),ROOT/'reports',ROOT/'models')

TASKS={'ingest':ingest,'validate':validate,'prepare':prepare,'transform':transform,'features':features,'train':train,'evaluate':evaluate,'track':track}
