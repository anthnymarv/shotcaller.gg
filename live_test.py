#!/usr/bin/env python3
"""
Live test the dragon/baron model during a game.

Option 1 - Quick prediction (you type numbers when drake/baron is up):
  python live_test.py

Option 2 - Log predictions to CSV, fill "got_it" after the game to measure accuracy:
  python live_test.py --log
  Then open data/live_test_log.csv, set got_it=1 if you got the objective else 0.

All inputs are player-visible (scoreboard, Tab, map). No Riot API during game.
"""
import os
import sys
import csv
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LOG_FILE = os.path.join(SCRIPT_DIR, 'data', 'live_test_log.csv')


def get_default_state():
    """Sensible defaults; you override when prompted."""
    return {
        'level_diff': 0,
        'ally_alive': 5,
        'enemy_alive': 5,
        'game_time_minutes': 15,
        'cs_per_min_diff': 0,
        'ally_cs_per_min': 25,
        'enemy_cs_per_min': 25,
        'ally_completed_items': 8,
        'enemy_completed_items': 8,
        'item_advantage': 0,
        'kills_last_2min_ally': 0,
        'kills_last_2min_enemy': 0,
        'kill_diff_recent': 0,
        'ally_towers': 9,
        'enemy_towers': 9,
        'tower_diff': 0,
        'time_since_last_fight': 120,
        'is_infernal': 0,
        'is_mountain': 0,
        'is_ocean': 0,
        'is_cloud': 0,
        'is_hextech': 0,
        'is_chemtech': 0,
        'is_baron': 0,
    }


def prompt_state():
    """Quick prompts for the most important fields; rest use defaults."""
    s = get_default_state()
    print("Enter numbers when drake/baron is up (or press Enter for default).\n")
    try:
        gt = input("Game time (minutes)? [15] ").strip() or "15"
        s['game_time_minutes'] = int(gt)
        ld = input("Level diff (your team - enemy, e.g. 1 or -1)? [0] ").strip() or "0"
        s['level_diff'] = int(ld)
        aa = input("Allies alive (0-5)? [5] ").strip() or "5"
        s['ally_alive'] = int(aa)
        ea = input("Enemies alive (0-5)? [5] ").strip() or "5"
        s['enemy_alive'] = int(ea)
        adv = input("Item advantage (your completed - theirs)? [0] ").strip() or "0"
        s['item_advantage'] = int(adv)
        s['ally_completed_items'] = 8 + max(0, s['item_advantage'])
        s['enemy_completed_items'] = s['ally_completed_items'] - s['item_advantage']
        td = input("Tower diff (ally - enemy standing)? [0] ").strip() or "0"
        s['tower_diff'] = int(td)
        obj = input("Objective: 1=Infernal 2=Mountain 3=Ocean 4=Cloud 5=Hextech 6=Chemtech 7=Baron [1] ").strip() or "1"
        o = int(obj)
        s['is_infernal']=s['is_mountain']=s['is_ocean']=s['is_cloud']=s['is_hextech']=s['is_chemtech']=s['is_baron']=0
        if o == 1: s['is_infernal']=1
        elif o == 2: s['is_mountain']=1
        elif o == 3: s['is_ocean']=1
        elif o == 4: s['is_cloud']=1
        elif o == 5: s['is_hextech']=1
        elif o == 6: s['is_chemtech']=1
        elif o == 7: s['is_baron']=1
    except (ValueError, KeyboardInterrupt):
        pass
    return s


def main():
    import argparse
    p = argparse.ArgumentParser(description='Live test objective model')
    p.add_argument('--log', action='store_true', help='Append prediction to data/live_test_log.csv')
    p.add_argument('--no-prompt', action='store_true', help='Use defaults only (no input)')
    args = p.parse_args()

    from shot_caller import ShotCaller
    sc = ShotCaller()

    state = get_default_state() if args.no_prompt else prompt_state()
    out = sc.should_take_objective(state)

    print("\n" + "="*50)
    print("SHOT CALLER")
    print("="*50)
    print(f"  Success probability: {out['probability']:.1%}")
    print(f"  Recommendation:      {out['recommendation']}")
    print(f"  Risk:                {out['risk']}")
    print("="*50)

    if args.log:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['time', 'game_time_min', 'objective', 'probability', 'recommendation', 'got_it'])
            if not file_exists:
                w.writeheader()
            obj = 'baron' if state.get('is_baron') else 'drake'
            w.writerow({
                'time': datetime.now().isoformat(),
                'game_time_min': state.get('game_time_minutes'),
                'objective': obj,
                'probability': out['probability'],
                'recommendation': out['recommendation'],
                'got_it': '',  # fill after game: 1 if you got it, 0 if not
            })
        print(f"\nLogged to {LOG_FILE} — fill 'got_it' (1/0) after the game to measure accuracy.")


if __name__ == '__main__':
    main()
