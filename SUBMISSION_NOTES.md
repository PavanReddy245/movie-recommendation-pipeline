# Submission completion notes

## Completed in the local project

- All eight pipeline stages are implemented and run end-to-end.
- Reusable engineered user/item/interaction features are generated for training and inference.
- The personalized SVD ranker consumes engineered features (preferred genre and item popularity) and uses a genre-aware re-ranking step.
- A CLI prediction example creates top-K recommendations for a supplied user.
- Local JSON experiment tracking records parameters, metrics and model artifacts. MLflow integration is implemented as an optional backend.
- A scheduled Python runner, cron instructions, an eight-task Airflow DAG, and a Docker Compose Airflow environment are included.
- Unit/integration tests and GitHub Actions CI configuration are included.

## Environment-dependent submission evidence

1. **Airflow successful-run screenshot:** Docker/Airflow was not available in the build environment. Start the included stack using `docker compose -f docker-compose.airflow.yaml up --build airflow`, trigger the DAG in the UI, wait for all eight tasks to succeed, and save the real Graph/Grid screenshot to `reports/airflow_successful_run.png`. No fabricated screenshot is included.
2. **GitHub repository URL:** Create a repository in your account and push using the commands in README. No remote or account authorization was available in this build, so the project has not been published.
3. **MLflow backend:** Local JSON experiment tracking was run and verified. MLflow is implemented as an optional backend but was not installed in this environment; install requirements and configure `MLFLOW_TRACKING_URI` if you want the MLflow UI evidence.
