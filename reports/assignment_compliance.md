# Assignment 2 requirement-by-requirement compliance

| Assignment requirement | Implemented location / evidence | Status |
|---|---|---|
| Ingest user details | `src/ingestion/ingest.py`, `data/raw/users.csv` | Complete |
| Ingest item details | `src/ingestion/ingest.py`, `data/raw/movies.csv` | Complete |
| Ingest interactions (ratings/views/clicks/purchases) | Ratings-based interactions in `data/raw/ratings.csv`; synthetic data documented | Complete |
| Automated missing-value checks | `src/validation/validate.py`, `data/errors/invalid_records.csv` | Complete |
| Duplicate checks | Metadata keys and normalized interaction timestamp keys | Complete |
| Incorrect data-type checks | Numeric IDs/ratings, age/year, string metadata and timestamp parsing | Complete |
| Invalid rating checks | Range 1–5 and non-numeric rating detection | Complete |
| Invalid timestamp checks | UTC timestamp parsing and quarantine | Complete |
| Referential integrity | Interaction user/item IDs checked against valid metadata | Complete |
| Invalid-record logging / error dataset | `data/errors/invalid_records.csv`, `validation_report.csv` | Complete |
| Cleaning, missing values, duplicates and standardized formats | `src/validation/validate.py`, UTC normalization | Complete |
| Time-based train/test split | `src/preparation/prepare.py`, 4,000 train / 1,000 test | Complete |
| User activity summary | `src/transformation/transform.py`, `user_activity_summary.csv` | Complete |
| Item popularity summary | `src/transformation/transform.py`, `item_popularity.csv` | Complete |
| User-item interaction matrix | `src/transformation/transform.py`, `user_item_matrix.csv` | Complete |
| User, item and interaction features documented | `src/features/build_features.py`, `data/features/feature_catalog.csv` | Complete |
| Features reusable for training and prediction | Shared feature definitions and `build_inference_features`; `recommend.py` | Complete |
| Popularity baseline | `PopularityRecommender` in `src/training/train_models.py` | Complete |
| Personalized recommender | Feature-aware SVD / hybrid ranker in `src/training/train_models.py` | Complete |
| Experiment parameters, metrics and artifacts tracked | `src/tracking.py`, `reports/experiment_tracking.json` with run ID and SHA-256 artifact manifest; optional MLflow | Complete via local JSON tracker; MLflow not installed in verification environment |
| Precision@K and Recall@K | `src/evaluation/evaluate.py`, K=5 | Complete |
| NDCG@K | Independent ideal ranking in `src/evaluation/evaluate.py` | Complete |
| Coverage and Diversity | Catalog coverage and genre pairwise diversity in `src/evaluation/evaluate.py` | Complete |
| RMSE and MAE for explicit ratings | SVD rating predictions in `src/evaluation/evaluate.py` | Complete |
| Results table comparing both models | `reports/model_comparison.csv` and final report | Complete |
| Workflow orchestration and task dependencies | `run_pipeline.py`, `pipeline_tasks.py`, eight-task Airflow DAG | Complete in code |
| Reproducible scheduled workflow | `scripts/run_scheduled.sh`, cron instructions | Complete |
| GitHub repository | CI workflow is included; project has not been pushed to user's GitHub | Pending user publishing action |
| Architecture diagram | `reports/architecture_diagram.png`, `.mmd`, README Mermaid diagram | Complete |
| Validation report/logs | `data/errors/validation_report.csv`, `logs/` | Complete |
| Feature catalog | `data/features/feature_catalog.csv` | Complete |
| Orchestration DAG/workflow file | `dags/recommendation_pipeline_dag.py` and `docker-compose.airflow.yaml` | Complete |
| Screenshot of successful Airflow run | Must be captured from actual Airflow UI after starting Docker stack | Pending environment execution; no fabricated screenshot |
| 2–3 page final report | `reports/final_report.pdf` | Complete |
| README and setup instructions | `README.md` | Complete |
| requirements file | `requirements.txt` | Complete |
| End-to-end run and automated tests | `logs/pipeline_run.txt`, `logs/test_run.txt`; 8 tests pass | Verified |

## Submission actions still requiring the user's accounts/environment

1. Start the supplied Airflow Docker Compose stack, trigger `movie_recommendation_end_to_end`, and capture the successful Graph/Grid view as `reports/airflow_successful_run.png`.
2. Create a GitHub repository, push this project, and replace the placeholder instructions in `README.md` with the real repository URL.
