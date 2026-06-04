import mlflow
import json
from pathlib import Path

REGISTRY = Path('experiments/registry.json')
# Use a local SQLite backend for MLflow tracking to avoid Windows file-store URI issues
mlflow_db = Path('mlflow.db').resolve()
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")

if not REGISTRY.exists():
    print('No registry found; nothing to log to MLflow')
    raise SystemExit(0)

registry = json.loads(REGISTRY.read_text())

for exp in registry:
    run_name = exp.get('id')
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('git_commit', exp.get('git_commit'))
        mlflow.set_tag('mode', exp.get('mode'))
        mlflow.set_tag('scope', exp.get('scope'))
        # log metrics
        metrics = exp.get('metrics', {})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
        # log artifacts if available
        mpath = Path(exp.get('model_path'))
        fpath = Path(exp.get('features_path'))
        if mpath.exists():
            mlflow.log_artifact(str(mpath), artifact_path='model')
        if fpath.exists():
            mlflow.log_artifact(str(fpath), artifact_path='features')

print('Logged registry to MLflow at', mlflow.get_tracking_uri())
