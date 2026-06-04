import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'experiments' / 'results'
REGISTRY = ROOT / 'experiments' / 'registry.json'

def run_training(mode='political', scope='all'):
    # map to train_lgbm.py args
    script = ROOT / 'models' / 'training' / 'train_lgbm.py'
    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")
    cmd = [sys.executable, str(script), '--mode', mode, '--variant', 'all_years' if scope=='all' else 'year2017']
    subprocess.check_call(cmd)

def register_experiment(mode='political', scope='all'):
    variant = 'all_years' if scope == 'all' else 'year2017'
    name = f"lgbm_{variant}_{mode}"
    result_path = RESULTS_DIR / f"{name}.json"
    metrics_path = Path('models') / 'artifacts' / f"{name}_metrics.json"
    features_path = Path('models') / 'artifacts' / f"{name}_features.json"
    model_path = Path('models') / f"lgbm_{variant}_{mode}.txt"
    script = ROOT / 'models' / 'training' / 'train_lgbm.py'

    with open(result_path) as f:
        results = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(features_path) as f:
        features = json.load(f)

    try:
        git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
    except Exception:
        git = None

    exp = {
        'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': git,
        'script': str(script),
        'name': name,
        'mode': mode,
        'scope': scope,
        'hypothesis': 'Automated run: baseline LGBM with selected features',
        'reasoning': 'Automated comparative experiment',
        'metrics': metrics,
        'results': results,
        'features_path': str(features_path),
        'model_path': str(model_path),
    }

    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text())
    else:
        registry = []
    registry.append(exp)
    REGISTRY.write_text(json.dumps(registry, indent=2))
    print('Experiment registered:', exp['id'])

def promote_champion():
    runner = ROOT / 'src' / 'experiment_runner.py'
    subprocess.check_call([sys.executable, str(runner)])

if __name__ == '__main__':
    # run one training, register, promote
    mode = sys.argv[1] if len(sys.argv) > 1 else 'political'
    scope = sys.argv[2] if len(sys.argv) > 2 else 'all'
    run_training(mode=mode, scope=scope)
    register_experiment(mode=mode, scope=scope)
    promote_champion()
