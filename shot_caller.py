#!/usr/bin/env python3
"""
Shot Caller — Deploy-ready objective predictor for players.

All inputs are player-visible only (scoreboard, Tab, map, kill feed).
Use this for in-game tools: "Should we take this drake/baron?"
Later: extend to recall timing and drake value for your team vs enemy.

Usage:
  from shot_caller import ShotCaller
  sc = ShotCaller()
  out = sc.should_take_objective(game_state_dict)
  # out = {'probability': 0.72, 'recommendation': 'GO', 'risk': 'LOW'}
"""

import os
import pickle
import pandas as pd
from typing import Dict, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'objective_model.pkl')


# Player-visible feature schema (for API/docs). All from in-game sight.
PLAYER_VISIBLE_SCHEMA = {
    'level_diff': 'Level advantage (your team - enemy). From scoreboard.',
    'ally_alive': 'Number of your team alive (0-5).',
    'enemy_alive': 'Number of enemy team alive (0-5).',
    'game_time_minutes': 'Current game time in minutes.',
    'cs_per_min_diff': 'CS per minute diff (ally - enemy). From scoreboard.',
    'ally_cs_per_min': 'Your team total CS / game minutes.',
    'enemy_cs_per_min': 'Enemy team total CS / game minutes.',
    'ally_completed_items': 'Completed items on your team (estimate from Tab).',
    'enemy_completed_items': 'Completed items on enemy team.',
    'item_advantage': 'ally_completed_items - enemy_completed_items.',
    'kills_last_2min_ally': 'Your team kills in last ~2 min (kill feed).',
    'kills_last_2min_enemy': 'Enemy kills in last ~2 min.',
    'kill_diff_recent': 'kills_last_2min_ally - kills_last_2min_enemy.',
    'ally_towers': 'Towers your team has left (e.g. 11 = none lost).',
    'enemy_towers': 'Towers enemy has left.',
    'tower_diff': 'ally_towers - enemy_towers.',
    'time_since_last_fight': 'Seconds since last big fight (estimate).',
    'is_infernal': 1 if current objective is Infernal drake else 0,
    'is_mountain': 1 if Mountain drake else 0,
    'is_ocean': 1 if Ocean drake else 0,
    'is_cloud': 1 if Cloud drake else 0,
    'is_hextech': 1 if Hextech drake else 0,
    'is_chemtech': 1 if Chemtech drake else 0,
    'is_baron': 1 if Baron else 0,
}


class ShotCaller:
    """
    Loads the trained objective model and exposes a single method:
    should_take_objective(game_state) -> probability, recommendation, risk.
    All inputs must be player-visible (no hidden data).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}. Run retrain.py first (or train_model.py)."
            )
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = list(data['feature_names'])
        self.accuracy = data.get('accuracy')
        self.roc_auc = data.get('roc_auc')
    
    def should_take_objective(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether your team should take the current objective (drake/baron).
        
        Args:
            game_state: Dict of player-visible features. Can be partial;
                        missing features are filled with 0.
        
        Returns:
            {
                'probability': float in [0, 1],
                'recommendation': 'GO' | 'RISKY' | 'SKIP',
                'risk': 'LOW' | 'MEDIUM' | 'HIGH',
            }
        """
        # Build row in model order; missing keys -> 0
        row = {fn: game_state.get(fn, 0) for fn in self.feature_names}
        X = pd.DataFrame([row])[self.feature_names]
        prob = float(self.model.predict_proba(X)[0][1])
        
        if prob > 0.6:
            recommendation = 'GO'
        elif prob > 0.4:
            recommendation = 'RISKY'
        else:
            recommendation = 'SKIP'
        
        if prob > 0.7:
            risk = 'LOW'
        elif prob > 0.5:
            risk = 'MEDIUM'
        else:
            risk = 'HIGH'
        
        return {
            'probability': round(prob, 3),
            'recommendation': recommendation,
            'risk': risk,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return accuracy and feature list (for deployment/UI)."""
        return {
            'accuracy': self.accuracy,
            'roc_auc': self.roc_auc,
            'feature_names': self.feature_names,
        }


def main():
    """CLI: run a single prediction with example state."""
    sc = ShotCaller()
    # Example: Mountain drake, slight lead, 20 min
    state = {
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
    out = sc.should_take_objective(state)
    print("Shot Caller — Objective decision (player-visible inputs only)")
    print("  Model accuracy:", sc.accuracy)
    print("  Example state: Mountain drake, 20 min, slight lead")
    print("  Probability:", out['probability'])
    print("  Recommendation:", out['recommendation'])
    print("  Risk:", out['risk'])


if __name__ == '__main__':
    main()
