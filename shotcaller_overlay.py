"""
shot_caller_overlay.py
----------------------
Always-on-top overlay that floats over League of Legends and shows
real-time GO / RISKY / NO GO recommendations.

Usage:
    python shot_caller_overlay.py

Requirements:
    pip install requests pandas xgboost
    tkinter is built into Python — no install needed.

Notes:
    - Run League in Borderless Windowed mode (not Fullscreen) for the overlay to show.
    - Drag the overlay by clicking and dragging anywhere on it.
    - The overlay auto-hides when no game is running and reappears when one starts.
"""

import os
import sys
import time
import pickle
import json
import csv
import threading
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import tkinter as tk
from tkinter import font as tkfont

import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

LIVE_API_BASE    = 'https://127.0.0.1:2999'
POLL_INTERVAL    = 5
GO_THRESHOLD     = 0.62
RISKY_THRESHOLD  = 0.42

_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, 'models', 'objective_model.pkl')
LOG_PATH   = os.path.join(_DIR, 'data', 'live_test_log.csv')

LOG_HEADER = [
    'timestamp', 'game_time_minutes', 'objective_type',
    'probability', 'recommendation',
    'level_diff', 'ally_alive', 'enemy_alive',
    'item_advantage', 'kill_diff_recent', 'tower_diff',
    'got_it',
]

DRAKE_MAP = {
    'fire': 'is_infernal', 'infernal': 'is_infernal',
    'earth': 'is_mountain', 'mountain': 'is_mountain',
    'water': 'is_ocean',    'ocean':    'is_ocean',
    'air':   'is_cloud',    'cloud':    'is_cloud',
    'hextech': 'is_hextech', 'chemtech': 'is_chemtech',
}

# ------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------

BG          = '#0a0e14'
BG_CARD     = '#111720'
BORDER      = '#1e2d3d'
GO_COLOR    = '#00ff87'
RISKY_COLOR = '#ffb627'
NOGO_COLOR  = '#ff3d5a'
TEXT_DIM    = '#4a6080'
TEXT_MAIN   = '#c8d8e8'
TEXT_BRIGHT = '#ffffff'
ACCENT      = '#1e90ff'


# ==================================================================
# Model
# ==================================================================

def load_model(path: str) -> Tuple[object, List[str]]:
    if not os.path.exists(path):
        print(f"[ERROR] Model not found at {path}")
        sys.exit(1)
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['feature_names']


# ==================================================================
# Live Client API
# ==================================================================

def _get(endpoint: str) -> Optional[dict]:
    try:
        r = requests.get(f'{LIVE_API_BASE}{endpoint}', verify=False, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def build_game_state(feature_names: List[str]) -> Optional[Dict]:
    stats   = _get('/liveclientdata/gamestats')
    players = _get('/liveclientdata/playerlist')
    events  = _get('/liveclientdata/eventdata')
    active  = _get('/liveclientdata/activeplayer')

    if not stats or not players or not events:
        return None

    game_time_sec = stats.get('gameTime', 0)
    game_time_min = game_time_sec / 60.0

    local_summoner = active.get('summonerName', '') if active else ''
    local_team = 'ORDER'
    for p in players:
        if p.get('summonerName') == local_summoner:
            local_team = p.get('team', 'ORDER')
            break

    ally_tag  = local_team
    enemy_tag = 'CHAOS' if ally_tag == 'ORDER' else 'ORDER'

    ally  = {'level': 0, 'alive': 0, 'cs': 0, 'items': 0}
    enemy = {'level': 0, 'alive': 0, 'cs': 0, 'items': 0}

    for p in players:
        b = ally if p.get('team') == ally_tag else enemy
        b['level'] += p.get('level', 1)
        b['cs']    += p.get('scores', {}).get('creepScore', 0)
        if not (p.get('isDead', False) or p.get('respawnTimer', 0) > 0):
            b['alive'] += 1
        b['items'] += sum(1 for it in p.get('items', []) if it.get('price', 0) >= 2500)

    elapsed = max(1, game_time_min)
    ally_cs_pm  = ally['cs']  / elapsed
    enemy_cs_pm = enemy['cs'] / elapsed

    all_events  = events.get('Events', [])
    cutoff      = game_time_sec - 120
    team_lookup = {p['summonerName']: p.get('team', 'ORDER') for p in players}

    ally_kills = enemy_kills = 0
    for ev in all_events:
        if ev.get('EventName') != 'ChampionKill' or ev.get('EventTime', 0) < cutoff:
            continue
        t = team_lookup.get(ev.get('KillerName', ''))
        if t == ally_tag:   ally_kills  += 1
        elif t == enemy_tag: enemy_kills += 1

    ally_towers = enemy_towers = 11
    for ev in all_events:
        if ev.get('EventName') != 'TurretKilled':
            continue
        name = ev.get('TurretKilled', '')
        if 'T1' in name:
            if ally_tag == 'ORDER': ally_towers  -= 1
            else:                   enemy_towers -= 1
        elif 'T2' in name:
            if ally_tag == 'CHAOS': ally_towers  -= 1
            else:                   enemy_towers -= 1

    fight_window = []
    for ev in reversed(all_events):
        if ev.get('EventName') != 'ChampionKill': continue
        t = ev.get('EventTime', 0)
        if not fight_window or fight_window[-1] - t < 10:
            fight_window.append(t)
        else:
            break
    time_since_fight = int(game_time_sec - fight_window[-1]) if len(fight_window) >= 2 else 300

    drake_flags = {k: 0 for k in ['is_infernal','is_mountain','is_ocean','is_cloud','is_hextech','is_chemtech']}
    is_baron = 0
    for ev in reversed(all_events):
        if ev.get('EventName') == 'DragonKill':
            key = DRAKE_MAP.get(ev.get('DragonType', '').lower())
            if key and key in drake_flags:
                drake_flags[key] = 1
            break
        if ev.get('EventName') == 'BaronKill':
            is_baron = 1
            break

    state = {
        'level_diff':            ally['level']  - enemy['level'],
        'ally_alive':            ally['alive'],
        'enemy_alive':           enemy['alive'],
        'game_time_minutes':     game_time_min,
        'cs_per_min_diff':       ally_cs_pm - enemy_cs_pm,
        'ally_cs_per_min':       ally_cs_pm,
        'enemy_cs_per_min':      enemy_cs_pm,
        'ally_completed_items':  ally['items'],
        'enemy_completed_items': enemy['items'],
        'item_advantage':        ally['items']  - enemy['items'],
        'kills_last_2min_ally':  ally_kills,
        'kills_last_2min_enemy': enemy_kills,
        'kill_diff_recent':      ally_kills - enemy_kills,
        'ally_towers':           max(0, ally_towers),
        'enemy_towers':          max(0, enemy_towers),
        'tower_diff':            ally_towers - enemy_towers,
        'time_since_last_fight': time_since_fight,
        'is_baron':              is_baron,
        **drake_flags,
    }
    for fn in feature_names:
        if fn not in state:
            state[fn] = 0
    return state


def predict(model, feature_names, state) -> float:
    X = pd.DataFrame([state])[feature_names]
    return float(model.predict_proba(X)[0][1])


def get_recommendation(prob: float) -> Tuple[str, str]:
    if prob >= GO_THRESHOLD:
        return 'GO', GO_COLOR
    elif prob >= RISKY_THRESHOLD:
        return 'RISKY', RISKY_COLOR
    return 'NO GO', NOGO_COLOR


# ==================================================================
# Logger
# ==================================================================

def init_log(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=LOG_HEADER).writeheader()


def log_prediction(path, state, prob, rec, obj_type):
    row = {
        'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'game_time_minutes': round(state.get('game_time_minutes', 0), 1),
        'objective_type':    obj_type,
        'probability':       round(prob, 4),
        'recommendation':    rec,
        'level_diff':        state.get('level_diff', 0),
        'ally_alive':        state.get('ally_alive', 0),
        'enemy_alive':       state.get('enemy_alive', 0),
        'item_advantage':    state.get('item_advantage', 0),
        'kill_diff_recent':  state.get('kill_diff_recent', 0),
        'tower_diff':        state.get('tower_diff', 0),
        'got_it':            '',
    }
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=LOG_HEADER).writerow(row)


# ==================================================================
# Overlay UI
# ==================================================================

class ShotCallerOverlay:
    def __init__(self, model, feature_names):
        self.model         = model
        self.feature_names = feature_names
        self.last_logged   = -1
        self._drag_x       = 0
        self._drag_y       = 0

        self.root = tk.Tk()
        self.root.overrideredirect(True)        # no title bar
        self.root.attributes('-topmost', True)  # always on top
        self.root.attributes('-alpha', 0.92)    # slight transparency
        self.root.configure(bg=BG)

        # Start position — top right corner
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f'230x260+{sw - 250}+20')

        self._build_ui()
        self._bind_drag()

        # Start polling in background thread
        self.running = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self.root

        # Outer border frame
        outer = tk.Frame(root, bg=BORDER, padx=1, pady=1)
        outer.pack(fill='both', expand=True)

        inner = tk.Frame(outer, bg=BG, padx=12, pady=10)
        inner.pack(fill='both', expand=True)

        # Header row
        header = tk.Frame(inner, bg=BG)
        header.pack(fill='x', pady=(0, 6))

        tk.Label(header, text='SHOT', bg=BG, fg=ACCENT,
                 font=('Courier', 11, 'bold')).pack(side='left')
        tk.Label(header, text='CALLER', bg=BG, fg=TEXT_BRIGHT,
                 font=('Courier', 11, 'bold')).pack(side='left')

        self.lbl_time = tk.Label(header, text='--:--', bg=BG, fg=TEXT_DIM,
                                  font=('Courier', 9))
        self.lbl_time.pack(side='right')

        # Divider
        tk.Frame(inner, bg=BORDER, height=1).pack(fill='x', pady=(0, 10))

        # Objective label
        self.lbl_obj = tk.Label(inner, text='WAITING FOR GAME',
                                bg=BG, fg=TEXT_DIM,
                                font=('Courier', 8, 'bold'),
                                anchor='center')
        self.lbl_obj.pack(fill='x')

        # Big recommendation label
        self.lbl_rec = tk.Label(inner, text='—',
                                bg=BG, fg=TEXT_DIM,
                                font=('Courier', 26, 'bold'),
                                anchor='center')
        self.lbl_rec.pack(fill='x', pady=(4, 0))

        # Probability bar
        bar_frame = tk.Frame(inner, bg=BG)
        bar_frame.pack(fill='x', pady=(6, 0))

        self.lbl_pct = tk.Label(bar_frame, text='  —  ', bg=BG, fg=TEXT_DIM,
                                font=('Courier', 10, 'bold'))
        self.lbl_pct.pack(side='right')

        self.bar_bg = tk.Frame(bar_frame, bg=BORDER, height=6)
        self.bar_bg.pack(side='left', fill='x', expand=True, pady=6)

        self.bar_fill = tk.Frame(self.bar_bg, bg=TEXT_DIM, height=6, width=0)
        self.bar_fill.place(x=0, y=0, relheight=1)

        # Divider
        tk.Frame(inner, bg=BORDER, height=1).pack(fill='x', pady=(8, 6))

        # Stats grid
        stats_frame = tk.Frame(inner, bg=BG)
        stats_frame.pack(fill='x')

        self.stat_labels = {}
        stats = [
            ('alive',  'ALIVE'),
            ('levels', 'LEVELS'),
            ('items',  'ITEMS'),
            ('kills',  'KILLS 2m'),
            ('towers', 'TOWERS'),
        ]

        for i, (key, label) in enumerate(stats):
            row = tk.Frame(stats_frame, bg=BG)
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label, bg=BG, fg=TEXT_DIM,
                     font=('Courier', 7), width=8, anchor='w').pack(side='left')
            val = tk.Label(row, text='—', bg=BG, fg=TEXT_MAIN,
                           font=('Courier', 7, 'bold'), anchor='e')
            val.pack(side='right')
            self.stat_labels[key] = val

        # Close button
        tk.Frame(inner, bg=BORDER, height=1).pack(fill='x', pady=(8, 4))
        close = tk.Label(inner, text='✕ close', bg=BG, fg=TEXT_DIM,
                         font=('Courier', 7), cursor='hand2')
        close.pack(anchor='e')
        close.bind('<Button-1>', lambda e: self.quit())

    # ------------------------------------------------------------------
    # Drag to move
    # ------------------------------------------------------------------

    def _bind_drag(self):
        self.root.bind('<ButtonPress-1>',   self._on_drag_start)
        self.root.bind('<B1-Motion>',       self._on_drag_move)

    def _on_drag_start(self, e):
        self._drag_x = e.x
        self._drag_y = e.y

    def _on_drag_move(self, e):
        x = self.root.winfo_x() + (e.x - self._drag_x)
        y = self.root.winfo_y() + (e.y - self._drag_y)
        self.root.geometry(f'+{x}+{y}')

    # ------------------------------------------------------------------
    # Update UI (called from main thread via after())
    # ------------------------------------------------------------------

    def _update_ui(self, state: Optional[Dict], prob: Optional[float]):
        if state is None:
            self.lbl_obj.config(text='WAITING FOR GAME', fg=TEXT_DIM)
            self.lbl_rec.config(text='—', fg=TEXT_DIM)
            self.lbl_pct.config(text='  —  ', fg=TEXT_DIM)
            self.lbl_time.config(text='--:--')
            self.bar_fill.config(bg=TEXT_DIM, width=0)
            for lbl in self.stat_labels.values():
                lbl.config(text='—', fg=TEXT_DIM)
            return

        rec, color = get_recommendation(prob)
        obj_type   = 'BARON' if state.get('is_baron') else 'DRAGON'
        gm         = state.get('game_time_minutes', 0)
        mm, ss     = int(gm), int((gm % 1) * 60)

        self.lbl_time.config(text=f'{mm:02d}:{ss:02d}')
        self.lbl_obj.config(text=f'▶  {obj_type}', fg=ACCENT)
        self.lbl_rec.config(text=rec, fg=color)
        self.lbl_pct.config(text=f'{prob:.0%}', fg=color)

        # Progress bar
        bar_w = self.bar_bg.winfo_width()
        fill  = int(bar_w * prob)
        self.bar_fill.config(bg=color, width=max(2, fill))

        # Stat rows
        def fmt_diff(val):
            return f'+{val}' if val > 0 else str(val)

        a, e = state.get('ally_alive', 0), state.get('enemy_alive', 0)
        self.stat_labels['alive'].config(
            text=f'{a}v{e}',
            fg=GO_COLOR if a > e else NOGO_COLOR if a < e else TEXT_MAIN
        )

        ld = state.get('level_diff', 0)
        self.stat_labels['levels'].config(
            text=fmt_diff(ld),
            fg=GO_COLOR if ld > 0 else NOGO_COLOR if ld < 0 else TEXT_MAIN
        )

        ia = state.get('item_advantage', 0)
        self.stat_labels['items'].config(
            text=fmt_diff(ia),
            fg=GO_COLOR if ia > 0 else NOGO_COLOR if ia < 0 else TEXT_MAIN
        )

        kd = state.get('kill_diff_recent', 0)
        self.stat_labels['kills'].config(
            text=fmt_diff(kd),
            fg=GO_COLOR if kd > 0 else NOGO_COLOR if kd < 0 else TEXT_MAIN
        )

        td = state.get('tower_diff', 0)
        self.stat_labels['towers'].config(
            text=fmt_diff(td),
            fg=GO_COLOR if td > 0 else NOGO_COLOR if td < 0 else TEXT_MAIN
        )

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _poll_loop(self):
        init_log(LOG_PATH)
        while self.running:
            try:
                state = build_game_state(self.feature_names)
                if state:
                    prob = predict(self.model, self.feature_names, state)
                    rec, _ = get_recommendation(prob)
                    obj_type = 'BARON' if state.get('is_baron') else 'DRAGON'

                    # Log once per minute
                    cur_min = int(state.get('game_time_minutes', 0))
                    if cur_min != self.last_logged and state.get('game_time_minutes', 0) >= 5:
                        log_prediction(LOG_PATH, state, prob, rec, obj_type)
                        self.last_logged = cur_min

                    self.root.after(0, self._update_ui, state, prob)
                else:
                    self.root.after(0, self._update_ui, None, None)

            except Exception as e:
                print(f'[poll error] {e}')

            time.sleep(POLL_INTERVAL)

    def quit(self):
        self.running = False
        self.root.destroy()


# ==================================================================
# Main
# ==================================================================

if __name__ == '__main__':
    print('Loading Shot Caller...')
    model, feature_names = load_model(MODEL_PATH)
    print('Model loaded. Starting overlay...')
    print('League must be in Borderless Windowed mode.')
    print('Drag the overlay to reposition it.')
    ShotCallerOverlay(model, feature_names)