import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'experiments' / 'results'
REGISTRY = ROOT / 'experiments' / 'registry.json'

def register_tuned(name, variant, mode):
    result_path = RESULTS_DIR / f"{name}.json"
    metrics_path = Path('models') / 'artifacts' / f"{name}_metrics.json"
    features_path = Path('models') / 'artifacts' / f"{name}_features.json"
    model_path = Path('models') / f"{name}.txt"
    if not result_path.exists():
        print('Missing result:', result_path)
        return
    with open(result_path) as f:
        results = json.load(f)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    try:
        git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
    except Exception:
        git = None
    exp = {
        'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': git,
        'script': 'models/training/train_lgbm_tune.py',
        'name': name,
        'mode': mode,
        'scope': variant,
        'hypothesis': f'Tuned LGBM: lr variant',
        'reasoning': 'Automated hyperparameter exploration',
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
    print('Registered tuned experiment:', exp['id'])
    # promote champion
    runner = ROOT / 'src' / 'experiment_runner.py'
    subprocess.check_call([sys.executable, str(runner)])

def main():
    candidates = [
        {'variant': 'all_years', 'mode': 'political', 'lr': 0.08, 'nl': 128},
        {'variant': 'all_years', 'mode': 'base', 'lr': 0.08, 'nl': 128},
    ]
    for c in candidates:
        name = f"lgbm_tuned_{c['variant']}_{c['mode']}_lr{c['lr']}_nl{c['nl']}"
        cmd = [sys.executable, str(ROOT / 'models' / 'training' / 'train_lgbm_tune.py'), '--variant', c['variant'], '--mode', c['mode'], '--learning_rate', str(c['lr']), '--num_leaves', str(c['nl'])]
        print('Running:', cmd)
        subprocess.check_call(cmd)
        register_tuned(name, c['variant'], c['mode'])

if __name__ == '__main__':
    main()
