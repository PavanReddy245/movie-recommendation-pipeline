"""Portable experiment tracking with optional MLflow backend."""
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone


def _clean(value):
    if value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return value.item() if hasattr(value, "item") else value


def _artifact_record(path, name=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "name": name or path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": digest,
    }


def log_experiment(metrics, config, report_dir, model_dir=None):
    """Persist a uniquely identified run, parameters, metrics and artifact manifest."""
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    records = [{k: _clean(v) for k, v in row.items()} for row in metrics.to_dict(orient="records")]

    candidates = []
    if model_dir:
        candidates.extend(Path(model_dir) / name for name in ("popularity_model.joblib", "svd_model.joblib"))
    candidates.extend(out / name for name in ("model_comparison.csv",))
    feature_dir = out.parent / "data" / "features"
    if feature_dir.exists():
        candidates.extend(sorted(p for p in feature_dir.iterdir() if p.is_file()))
    artifacts = [_artifact_record(path) for path in candidates if path.exists()]

    payload = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "movie-recommendation-assignment",
        "parameters": {k: _clean(v) for k, v in config.items()},
        "metrics": records,
        "artifacts": artifacts,
        "tracking_backend": "local_json",
        "mlflow_status": "not_attempted",
    }
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(payload["experiment_name"])
        with mlflow.start_run(run_name="movie-recommendation-pipeline") as run:
            payload["mlflow_run_id"] = run.info.run_id
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
            for record in records:
                model_key = str(record["model"]).lower().replace(" ", "_")
                for key, value in record.items():
                    if key != "model" and isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
                        mlflow.log_metric(f"{model_key}_{key}", float(value))
            for artifact in artifacts:
                mlflow.log_artifact(artifact["path"], artifact_path="artifacts")
        payload["tracking_backend"] = "mlflow+local_json"
        payload["mlflow_status"] = "logged_parameters_metrics_and_artifacts"
    except ImportError:
        payload["mlflow_status"] = "not_installed_local_json_tracking_used"
    except Exception as exc:
        payload["mlflow_status"] = f"local_json_fallback: {type(exc).__name__}: {exc}"

    (out / "experiment_tracking.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
