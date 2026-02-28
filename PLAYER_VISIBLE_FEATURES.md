# Player-visible features (deployment scope)

All data used for training and for the **Shot Caller** predictor is restricted to what a player can see in-game. No hidden or API-only data.

## In scope (use these for shot caller / future macro assistant)

| Feature | Where player sees it |
|--------|-----------------------|
| Level diff (team vs enemy) | Scoreboard (Tab) |
| Ally / enemy alive count | Map, team frames |
| Game time | In-game clock |
| CS per minute (team totals) | Scoreboard, optional apps |
| Completed items (count per team) | Tab (inspect items) |
| Kills in last ~2 minutes | Kill feed |
| Towers standing (per team) | Map (structures) |
| Time since last fight | Game sense / timestamps |
| Current objective type | Drake/baron icon, map |

## Not in scope (do not use for deployed predictor)

- Exact gold values (not shown to enemies; we use item counts instead)
- Exact cooldowns the enemy hasn’t revealed
- Fog-of-war or server-only state

## Retrain after a new patch

1. Update data: `python retrain.py` (or `--collect-only` then `--train-only`).
2. Check accuracy in the retrain output; if it’s acceptable, the new model is ready to ship.
3. Use `shot_caller.ShotCaller().should_take_objective(game_state)` for “take this objective?” in your app.

## Future shot-caller extensions

- **When to recall** — train on replay data with “good recall” labels.
- **Drake value for your team vs enemy** — combine with `drake_decision_system` and champion–drake synergy (e.g. `champion_dragon_analyzer`) so the shot caller can say “this drake is high value for us / for them.”
