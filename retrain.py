#!/usr/bin/env python3
"""
Retrain workflow for new patch/season.
Run this after a new League patch to collect fresh data and retrain the objective model.
  python retrain.py              # collect new data + train
  python retrain.py --collect-only   # only collect data
  python retrain.py --train-only     # only train (use existing data)
  python retrain.py --matches 100    # collect from 100 matches (default 80)
"""
import os
import sys
import shutil
from datetime import datetime

# Unbuffered stdout so output shows immediately when run from launcher or IDE
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'objective_training_data.csv')
DATA_BACKUP_DIR = os.path.join(SCRIPT_DIR, 'data', 'backups')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'models', 'objective_model.pkl')


def backup_data():
    """Backup current training data before overwriting (optional)."""
    if not os.path.exists(DATA_FILE):
        return
    os.makedirs(DATA_BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    backup_path = os.path.join(DATA_BACKUP_DIR, f'objective_training_data_{stamp}.csv')
    shutil.copy2(DATA_FILE, backup_path)
    print(f"  Backed up data to {os.path.basename(backup_path)}", flush=True)


def collect_data(num_matches: int = 80):
    """Collect fresh match data from Riot API (player-visible features only)."""
    print("="*60, flush=True)
    print("STEP 1: COLLECTING FRESH DATA (new patch/season)", flush=True)
    print("="*60, flush=True)
    
    # Load API key from api.env
    api_env_file = os.path.join(SCRIPT_DIR, 'api.env')
    API_KEY = None
    if os.path.exists(api_env_file):
        with open(api_env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('RIOT_API_KEY='):
                    API_KEY = line.split('=', 1)[1].strip()
                    break
                if line.startswith('RGAPI-'):
                    API_KEY = line
                    break
    if not API_KEY:
        API_KEY = os.environ.get('RIOT_API_KEY')
    if not API_KEY:
        print("ERROR: RIOT_API_KEY not found. Set it in api.env or environment.", flush=True)
        return False
    
    print("  Loading collector and starting API requests...", flush=True)
    from matchDataCollector import RiotDataCollector
    collector = RiotDataCollector(API_KEY)
    df = collector.collect_training_data(num_matches=num_matches, output_file=DATA_FILE)
    
    if len(df) == 0:
        print("ERROR: No data collected. Check API key and rate limits.")
        return False
    
    print(f"\n  Collected {len(df)} samples. Success rate: {df['objective_secured'].mean():.1%}", flush=True)
    return True


def train_model():
    """Train the objective model and return accuracy."""
    print("\n" + "="*60, flush=True)
    print("STEP 2: TRAINING MODEL", flush=True)
    print("="*60, flush=True)
    
    if not os.path.exists(DATA_FILE):
        print("ERROR: No data file. Run with --collect-only first or run without --train-only.")
        return None
    
    sys.path.insert(0, SCRIPT_DIR)
    from train_model import train_objective_model
    
    model, _ = train_objective_model(DATA_FILE)
    with open(MODEL_FILE, 'rb') as f:
        import pickle
        data = pickle.load(f)
    accuracy = data.get('accuracy', 0)
    roc_auc = data.get('roc_auc', 0)
        print(f"\n  Model saved. Accuracy: {accuracy:.1%}  ROC-AUC: {roc_auc:.3f}", flush=True)
    return accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Retrain objective model for new patch/season')
    parser.add_argument('--collect-only', action='store_true', help='Only collect data, do not train')
    parser.add_argument('--train-only', action='store_true', help='Only train on existing data')
    parser.add_argument('--matches', type=int, default=80, help='Number of matches to collect (default 80)')
    parser.add_argument('--no-backup', action='store_true', help='Do not backup existing data before collect')
    args = parser.parse_args()
    
    do_collect = not args.train_only
    do_train = not args.collect_only
    
    if do_collect:
        if not args.no_backup:
            backup_data()
        ok = collect_data(num_matches=args.matches)
        if not ok:
            sys.exit(1)
    
    if do_train:
        acc = train_model()
        if acc is None:
            sys.exit(1)
        print("\n" + "="*60, flush=True)
        print("RETRAIN COMPLETE — Ready for deployment (shot_caller.py)", flush=True)
        print("="*60, flush=True)
        print(f"  Accuracy: {acc:.1%} — Use this to decide if the model is good enough to ship.", flush=True)


if __name__ == '__main__':
    main()
