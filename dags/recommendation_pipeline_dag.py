"""Eight-stage scheduled Airflow DAG; each task executes an independently rerunnable stage."""
from datetime import datetime, timedelta
from pathlib import Path
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = Path("/opt/recommendation-pipeline")
#PROJECT_DIR=Path(os.getenv('RECOMMENDATION_PROJECT_DIR', Path(__file__).resolve().parents[1]))
def task(stage):
    return BashOperator(
    task_id=stage,
    bash_command=f"cd {PROJECT_DIR} && python run_stage.py {stage}",
    retries=1,
    retry_delay=timedelta(minutes=5)
)
with DAG(dag_id='movie_recommendation_end_to_end',start_date=datetime(2026,1,1),schedule='@daily',catchup=False,default_args={'owner':'dm4ml','retries':1},tags=['dm4ml','recommendation']) as dag:
    ingest=task('ingest'); validate=task('validate'); prepare=task('prepare'); transform=task('transform'); features=task('features'); train=task('train'); evaluate=task('evaluate'); track=task('track')
    ingest >> validate >> prepare >> transform >> features >> train >> evaluate >> track
