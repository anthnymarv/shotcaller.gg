# Live testing & deployment (dragon/baron model)

Start with the dragon/baron model; later add recall timing, when to farm vs rotate, and item/objective value.

---

## 1. Getting game state during a live game

The model only uses **player-visible** data (no cheating). You have two ways to get that state:

### A. Manual entry (works anywhere)

When drake or baron is up, look at:

- **Scoreboard (Tab):** level diff, CS (you can estimate CS/min as total CS ÷ game time), items (count completed).
- **Map / frames:** allies alive, enemies alive, towers standing.
- **Kill feed:** rough count of kills in the last ~2 minutes.
- **Clock:** game time in minutes; time since last fight (guess).

Then either:

- Run **`python live_test.py`** and type the numbers when prompted, or
- Build a small overlay or web form that takes these fields and calls `ShotCaller().should_take_objective(state)` and shows **GO / RISKY / SKIP** and probability.

### B. Riot Live Client API (same machine as the game)

If your app runs on the **same PC** as the League client while a game is running, you can read live state (e.g. gold, levels, positions) from the client and map it to the model’s features.

- [Riot Developer Portal – Live Client API](https://developer.riotgames.com/docs/lol#live-client-api)  
- Requires the game to be running locally; your app talks to the client over a local endpoint.
- You’d pull the same numbers (level diff, alive counts, CS, etc.) from the API and pass them into `should_take_objective()`.

We don’t include Live Client code here; the model and `shot_caller` are the same either way.

---

## 2. Running a live test (quick check)

From `lol_drakes_project` (or use the root `retrain.py` launcher from repo root):

```bash
cd lol_drakes_project
python live_test.py
```

When drake/baron is up, enter the prompted values. The script prints:

- **Success probability**
- **Recommendation:** GO / RISKY / SKIP  
- **Risk:** LOW / MEDIUM / HIGH  

Use that to decide in-game and see if it “feels” right.

---

## 3. Measuring real accuracy (log predictions, fill outcome later)

To see if the model actually predicts success:

1. **Log each prediction:**
   ```bash
   python live_test.py --log
   ```
   Each run appends a row to `data/live_test_log.csv` with timestamp, game time, objective (drake/baron), probability, and recommendation.

2. **After the game**, open `data/live_test_log.csv` and set **`got_it`**:
   - `1` = your team got that objective  
   - `0` = you didn’t (enemy took it or you skipped)

3. **Compute accuracy** over many games (e.g. in a spreadsheet or a small script):  
   accuracy = (number of rows where recommendation was GO and got_it=1, or SKIP and got_it=0) / total, or simply “when we said GO, how often did we get it?”.

That gives you a real-world accuracy number to compare with the test-set accuracy from training.

---

## 4. Deploying for users

- **Back end:** Expose one endpoint that accepts a JSON **game_state** (same keys as in `PLAYER_VISIBLE_FEATURES.md`) and returns `ShotCaller().should_take_objective(game_state)` (probability, recommendation, risk). No Riot API key needed at request time.
- **Front end:** Overlay or second screen that either:
  - Lets the user type the numbers (like `live_test.py`), or  
  - Pulls state from the Live Client API (same machine only) and sends it to your backend.
- **Later:** Add “when to recall”, “farm vs rotate”, and drake/baron **value** for your team vs enemy (using your drake decision system + champion–drake synergy). Same idea: feed only player-visible state into the shot caller.

---

## 5. Challenger vs Grandmaster and sample size

- **Current behavior:** The collector uses **Grandmaster first** (10 players), then if it has fewer than 5 PUUIDs it also fetches **Challenger** (5 players). So you do get Challenger as a fallback; most of the time the bulk of matches are from Grandmaster.
- **If you want more Challenger:** In `matchDataCollector.py`, you can increase Challenger count (e.g. `get_challenger_players(count=10)`) or fetch Challenger in addition to GM (e.g. 10 GM + 10 Challenger) so more matches are from Challenger.
- **Sample size:** More data usually helps generalization; 150–300+ objective events is a good target. With 80 matches you often get a few hundred rows; 100–150 matches can get you 400–600+. Rate limits (and a 24h dev key) are the main limit, so run `retrain.py` with `--matches 100` or `--matches 150` when you can. If test accuracy is already high and stable, more samples give diminishing returns; if it’s still improving, keep adding data.

---

## Summary

| Goal | What to do |
|------|------------|
| Try it live once | `python live_test.py`, enter numbers when drake/baron is up |
| Measure real accuracy | `python live_test.py --log`, fill `got_it` in `data/live_test_log.csv` after each game |
| Deploy for users | Backend: one endpoint that takes game_state, returns shot_caller output. Front: overlay or form that sends state (manual or Live Client). |
| More Challenger data | Increase Challenger count or add Challenger in addition to GM in `matchDataCollector.py`. |
| More samples | Run `python retrain.py --matches 100` (or 150) when you can; aim for 300+ objective rows if possible. |

Start with the dragon/baron model and live_test + log; once that’s solid, add recall, farm/rotate, and objective value on top.
