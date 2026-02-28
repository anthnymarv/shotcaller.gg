import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import pickle
import os

def prepare_features(df):
    """Convert raw data to model features"""
    
    # Core features (player-visible only)
    feature_cols = [
        'level_diff',
        'ally_alive', 
        'enemy_alive',
        'game_time_minutes',
        'cs_per_min_diff',
        'ally_cs_per_min',
        'enemy_cs_per_min',
        'ally_completed_items',
        'enemy_completed_items',
        'item_advantage',
        'kills_last_2min_ally',
        'kills_last_2min_enemy', 
        'kill_diff_recent',
        'ally_towers',
        'enemy_towers',
        'tower_diff',
        'time_since_last_fight',
        # Drake type one-hot encoding
        'is_infernal',
        'is_mountain',
        'is_ocean',
        'is_cloud',
        'is_hextech',
        'is_chemtech',
        'is_baron'
    ]
    
    # Only use features that exist in the data
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print(f"   Warning: Missing features: {missing}")
    
    features = df[available_features].copy()
    
    # Fill missing values
    features = features.fillna(0)
    
    return features

def train_objective_model(data_file='data/objective_training_data.csv'):
    """Train XGBoost model on collected data"""
    
    print("="*60)
    print("TRAINING OBJECTIVE PREDICTION MODEL")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} samples")
    
    # Prepare features
    print("\n2. Preparing features...")
    X = prepare_features(df)
    y = df['objective_secured']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Feature names: {list(X.columns)}")
    
    # Check class balance
    success_rate = y.mean()
    print(f"\n3. Data balance:")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Failures: {len(y[y==0])} | Successes: {len(y[y==1])}")
    
    if success_rate < 0.3 or success_rate > 0.7:
        print("   ⚠️  WARNING: Imbalanced data. Consider collecting more matches.")
    
    # Split data
    print("\n4. Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train)} | Testing: {len(X_test)}")
    
    # Train model
    print("\n5. Training XGBoost model...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("   ✓ Training complete!")
    
    # Evaluate
    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   ROC-AUC: {roc_auc:.3f}")
    
    if accuracy < 0.6:
        print("   ⚠️  Low accuracy. Need more/better features.")
    elif accuracy > 0.85:
        print("   ⚠️  Very high accuracy. Possible overfitting - check for data leakage.")
    else:
        print("   ✓ Good performance!")
    
    # Feature importance
    print("\n7. Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.3f}")
    
    # Save model
    print("\n8. Saving model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, 'objective_model.pkl')
    
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': list(X.columns),
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }, f)
    
    print(f"   ✓ Model saved to {model_filename}")
    
    # Test prediction
    print("\n9. Test Prediction:")
    sample = X_test.iloc[0:1]
    prob = model.predict_proba(sample)[0][1]
    actual = y_test.iloc[0]
    
    print(f"   Predicted success probability: {prob:.1%}")
    print(f"   Actual outcome: {'SUCCESS' if actual == 1 else 'FAILURE'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_filename}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    
    return model, X.columns


def predict_new_game(model_file=None):
    """Example: Use trained model to predict a new game state (player-visible features only)."""
    if model_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(script_dir, 'models', 'objective_model.pkl')
    
    print("\n" + "="*60)
    print("TESTING PREDICTION ON NEW GAME")
    print("="*60)
    
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data['model']
    feature_names = saved_data['feature_names']
    
    # Example game state: only player-visible inputs (scoreboard, Tab, map, kill feed)
    # Match the schema from matchDataCollector / prepare_features
    game_state = {
        'level_diff': 1,
        'ally_alive': 5,
        'enemy_alive': 4,
        'game_time_minutes': 20,
        'cs_per_min_diff': 2.0,
        'ally_cs_per_min': 28,
        'enemy_cs_per_min': 26,
        'ally_completed_items': 12,
        'enemy_completed_items': 10,
        'item_advantage': 2,
        'kills_last_2min_ally': 1,
        'kills_last_2min_enemy': 0,
        'kill_diff_recent': 1,
        'ally_towers': 5,
        'enemy_towers': 4,
        'tower_diff': 1,
        'time_since_last_fight': 90,
        'is_infernal': 0,
        'is_mountain': 1,
        'is_ocean': 0,
        'is_cloud': 0,
        'is_hextech': 0,
        'is_chemtech': 0,
        'is_baron': 0,
    }
    # Ensure all features expected by model are present
    for fn in feature_names:
        if fn not in game_state:
            game_state[fn] = 0
    
    X_new = pd.DataFrame([game_state])[feature_names]
    prob = model.predict_proba(X_new)[0][1]
    prediction = model.predict(X_new)[0]
    
    print(f"\nGame State (player-visible):")
    print(f"  Level diff: {game_state['level_diff']}  Alive: {game_state['ally_alive']} vs {game_state['enemy_alive']}")
    print(f"  Game Time: {game_state['game_time_minutes']} min  Drake: MOUNTAIN  Baron: No")
    print(f"\nPrediction:")
    print(f"  Success Probability: {prob:.1%}")
    print(f"  Recommendation: {'GO FOR IT!' if prob > 0.6 else 'RISKY' if prob > 0.4 else 'DO NOT ATTEMPT'}")
    risk = "LOW" if prob > 0.7 else "MEDIUM" if prob > 0.5 else "HIGH"
    print(f"  Risk Level: {risk}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'objective_training_data.csv')
    model, features = train_objective_model(data_file)
    print("\n")
    predict_new_game()