# shotcaller.gg

Real-time objective decision tool for League of Legends. Pulls live game state from Riot's Local Client API, runs it through an XGBoost model trained on Grandmaster+ match data, and tells you whether to take the drake or baron — **GO**, **RISKY**, or **NO GO**.

---

## How It Works

```
Grandmaster+ Match Data (Riot API)
        ↓
Feature Extraction (game state at each objective)
        ↓
XGBoost Classifier (trained on 2000+ high-elo games)
        ↓
Live Client (polls localhost:2999 every 5s)
        ↓
Terminal Dashboard → GO / RISKY / NO GO
```

---

## Features Used (Player-Visible Only)

The model only uses information a player can actually see in-game — no hidden gold values or fog-of-war data.

- Alive counts (ally vs enemy)
- Level differential
- CS per minute (both teams)
- Estimated completed items
- Recent kills (last 2 minutes)
- Tower differential
- Time since last teamfight
- Drake type (infernal, mountain, ocean, cloud, hextech, chemtech)
- Baron vs dragon flag

---

## Project Structure

```
shotcaller.gg/
├── matchDataCollector.py       # Pulls Grandmaster/Challenger match data from Riot API
├── shot_caller_pipeline.py     # Cleans data, runs MI feature selection, trains LR baseline
├── train_model.py              # Trains XGBoost model and saves to models/
├── shot_caller.py              # Live client — runs during game, outputs recommendation
├── live_test.py                # Logs live predictions to CSV for accuracy tracking
├── retrain.py                  # Retrains model on new data
├── requirements.txt
└── website/                    # Landing page
```

---

## Setup

**Requirements:** Python 3.9+, pip

```bash
git clone https://github.com/anthnymarv/shotcaller.gg.git
cd shotcaller.gg
pip install -r requirements.txt
```

Create an `api.env` file in the project root (not committed — keep this private):
```
RIOT_API_KEY=RGAPI-your-key-here
```

---

## Collecting Data & Training

```bash
# 1. Collect match data from high-elo games
python matchDataCollector.py

# 2. Train the model
python train_model.py
```

This saves a trained model to `models/objective_model.pkl`.

---

## Running Live

Open League of Legends, then in a separate terminal:

```bash
python shot_caller.py
```

The dashboard updates every 5 seconds and logs every prediction to `data/live_test_log.csv`. After each game, fill in the `got_it` column (1 = secured, 0 = lost) to track real-world accuracy over time.

---

## Tech Stack

- **Data:** Riot Match v5 API + Timeline API
- **Model:** XGBoost classifier (trained on ~2000 Grandmaster+ games)
- **Live data:** Riot Live Client Data API (`localhost:2999`)
- **ML pipeline:** scikit-learn, pandas, numpy

---

## Authors

Built by two guys who want jobs and also play too much League.
