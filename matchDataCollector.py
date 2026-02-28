import requests
import json
import time
import random
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os


# --- Drake keyword -> canonical name lookup (replaces 20+ if/elif chain) ---
DRAKE_KEYWORDS = {
    'FIRE': 'INFERNAL', 'INFERNAL': 'INFERNAL',
    'EARTH': 'MOUNTAIN', 'MOUNTAIN': 'MOUNTAIN',
    'WATER': 'OCEAN',    'OCEAN':    'OCEAN',
    'AIR':   'CLOUD',    'CLOUD':    'CLOUD',
    'HEXTECH':  'HEXTECH',
    'CHEMTECH': 'CHEMTECH',
}


class RiotDataCollector:
    """
    Collects match data from Riot API and extracts training data
    for objective prediction models.
    """

    def __init__(self, api_key: str, region: str = 'na1'):
        self.api_key = api_key
        self.region = region
        self.base_url = f'https://{region}.api.riotgames.com'
        self.americas_url = 'https://americas.api.riotgames.com'
        self.headers = {'X-Riot-Token': api_key}
        self.rate_limit_delay = 2.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """
        Central request helper with adaptive 429 handling.
        Uses the Retry-After header when rate-limited instead of a fixed wait.
        Returns None on non-200/429 errors after one retry.
        """
        while True:
            time.sleep(self.rate_limit_delay)
            try:
                response = requests.get(url, headers=self.headers,
                                        params=params, timeout=10)
            except Exception as e:
                print(f"  Network error: {e}")
                return None

            if response.status_code == 200:
                return response

            if response.status_code == 429:
                wait = int(response.headers.get('Retry-After', 60))
                print(f"  Rate limited. Waiting {wait}s (from Retry-After header)...")
                time.sleep(wait)
                continue  # retry immediately after waiting

            if response.status_code == 403:
                print("  ERROR: API key invalid or expired! "
                      "Get a new key at https://developer.riotgames.com/")
                return None

            print(f"  HTTP {response.status_code} for {url}")
            return None

    def _estimate_respawn_time_ms(self, game_minute: int) -> int:
        """
        Approximate respawn timer in milliseconds.
        Real formula scales with level; this uses game time as a proxy.
        Range: ~7s early game up to ~60s late game.
        """
        seconds = min(7 + int(game_minute * 2.5), 60)
        return seconds * 1000

    def _get_league_players(self, tier: str,
                            queue: str = 'RANKED_SOLO_5x5',
                            count: int = 50) -> List[str]:
        """
        Generic helper for fetching players from a given tier
        (grandmaster / challenger / master).
        Replaces the duplicated get_grandmaster_players / get_challenger_players.
        """
        print(f"Fetching {tier} players from {self.region}...")
        url = f'{self.base_url}/lol/league/v4/{tier}leagues/by-queue/{queue}'

        response = self._request(url)
        if response is None:
            return []

        data = response.json()
        if 'entries' not in data or not data['entries']:
            print("No entries found.")
            return []

        entries = data['entries'][:count]
        print(f"Found {len(entries)} {tier} entries.")
        return self._convert_entries_to_puuids(entries, count)

    def _convert_entries_to_puuids(self, entries: List[Dict],
                                   max_count: int) -> List[str]:
        """Convert league entries to PUUIDs, handling both API formats."""
        puuids: List[str] = []
        failed_count = 0

        for entry in entries:
            if len(puuids) >= max_count:
                break

            # Newer API: puuid is already present
            if 'puuid' in entry:
                puuids.append(entry['puuid'])
                if len(puuids) % 5 == 0:
                    print(f"  Progress: {len(puuids)} PUUIDs collected")
                continue

            # Older API: resolve via summoner endpoint
            summoner_id = (entry.get('summonerId') or
                           entry.get('id') or
                           entry.get('encryptedSummonerId'))
            if not summoner_id:
                failed_count += 1
                continue

            summoner_url = f'{self.base_url}/lol/summoner/v4/summoners/{summoner_id}'
            resp = self._request(summoner_url)

            if resp is None:
                failed_count += 1
                if failed_count > 10:
                    print("  Too many failures — stopping.")
                    break
                continue

            summoner_data = resp.json()
            if 'puuid' in summoner_data:
                puuids.append(summoner_data['puuid'])
                if len(puuids) % 5 == 0:
                    print(f"  Progress: {len(puuids)} PUUIDs collected")
            else:
                failed_count += 1

        print(f"Collected {len(puuids)} PUUIDs (failed: {failed_count})")
        return puuids

    def _save_with_dedup(self, df: pd.DataFrame, output_file: str) -> pd.DataFrame:
        """
        Append new samples to an existing CSV (if present) and deduplicate.
        Extracted from collect_training_data to keep that method readable.
        """
        if not os.path.exists(output_file):
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} samples to {output_file}")
            return df

        try:
            existing_df = pd.read_csv(output_file)
            combined = pd.concat([existing_df, df], ignore_index=True)

            dedup_cols = ['match_id', 'timestamp_minutes', 'objective_type']
            if all(c in combined.columns for c in dedup_cols):
                combined = combined.drop_duplicates(subset=dedup_cols, keep='last')

            combined.to_csv(output_file, index=False)
            print(f"Appended to existing file. Total samples now: {len(combined)}")
            return combined

        except Exception as e:
            print(f"Warning: Could not read existing file ({e}), overwriting.")
            df.to_csv(output_file, index=False)
            return df

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_grandmaster_players(self, queue: str = 'RANKED_SOLO_5x5',
                                count: int = 50) -> List[str]:
        return self._get_league_players('grandmaster', queue, count)

    def get_challenger_players(self, queue: str = 'RANKED_SOLO_5x5',
                               count: int = 50) -> List[str]:
        return self._get_league_players('challenger', queue, count)

    def get_match_ids(self, puuids: List[str],
                      matches_per_player: int = 20) -> List[str]:
        """Get match IDs from multiple players (ranked solo queue only)."""
        all_match_ids: set = set()

        for i, puuid in enumerate(puuids):
            url = f'{self.americas_url}/lol/match/v5/matches/by-puuid/{puuid}/ids'
            params = {'start': 0, 'count': matches_per_player, 'queue': 420}

            response = self._request(url, params=params)
            if response is None:
                print(f"Player {i+1}/{len(puuids)}: skipped (request failed)")
                continue

            match_ids = response.json()
            all_match_ids.update(match_ids)
            print(f"Player {i+1}/{len(puuids)}: "
                  f"{len(match_ids)} matches. Total unique: {len(all_match_ids)}")

        return list(all_match_ids)

    def get_match_data(self, match_id: str) -> Optional[Dict]:
        """Fetch match details and timeline for a single match."""
        match_url = f'{self.americas_url}/lol/match/v5/matches/{match_id}'
        match_resp = self._request(match_url)
        if match_resp is None:
            return None

        timeline_url = f'{self.americas_url}/lol/match/v5/matches/{match_id}/timeline'
        timeline_resp = self._request(timeline_url)
        if timeline_resp is None:
            return None

        return {'match': match_resp.json(), 'timeline': timeline_resp.json()}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_objective_events(self, match_data: Dict) -> List[Dict]:
        """Extract dragon/baron kill events and build training samples."""
        training_samples: List[Dict] = []

        frames = match_data['timeline']['info']['frames']

        for frame_idx, frame in enumerate(frames):
            timestamp = frame['timestamp']
            minute = timestamp // 60000
            participant_frames = frame.get('participantFrames', {})

            for event in frame.get('events', []):
                if event.get('type') != 'ELITE_MONSTER_KILL':
                    continue

                monster_type = event.get('monsterType', '')
                if 'DRAGON' not in monster_type and monster_type != 'BARON_NASHOR':
                    continue

                game_state = self._extract_game_state(
                    frame_idx, frames, participant_frames, event
                )
                if game_state is None:
                    continue

                killer_team = self._get_killer_team(event, match_data['match'])

                # Randomly flip perspective 50/50 to avoid blue-side bias
                analyze_from_blue = random.choice([True, False])

                if analyze_from_blue:
                    game_state['objective_secured'] = 1 if killer_team == 100 else 0
                else:
                    # Red-side perspective: flip label and all directional features
                    game_state['objective_secured'] = 1 if killer_team == 200 else 0
                    game_state['level_diff']       = -game_state['level_diff']
                    game_state['cs_per_min_diff']  = -game_state['cs_per_min_diff']
                    game_state['item_advantage']   = -game_state['item_advantage']
                    game_state['kill_diff_recent'] = -game_state['kill_diff_recent']
                    game_state['tower_diff']       = -game_state['tower_diff']

                    # Swap ally/enemy pairs
                    for ally_key, enemy_key in [
                        ('ally_alive',           'enemy_alive'),
                        ('ally_cs_per_min',      'enemy_cs_per_min'),
                        ('ally_completed_items', 'enemy_completed_items'),
                        ('kills_last_2min_ally', 'kills_last_2min_enemy'),
                        ('ally_towers',          'enemy_towers'),
                    ]:
                        game_state[ally_key], game_state[enemy_key] = (
                            game_state[enemy_key], game_state[ally_key]
                        )

                game_state['match_id']           = match_data['match']['metadata']['matchId']
                game_state['timestamp_minutes']  = minute
                game_state['objective_type']     = monster_type
                training_samples.append(game_state)

        return training_samples

    def _extract_game_state(self, frame_idx: int, frames: List,
                            participant_frames: Dict,
                            event: Dict) -> Optional[Dict]:
        """Extract feature vector from game state just before an objective kill."""
        if frame_idx == 0:
            return None

        prev_frame = frames[frame_idx - 1]
        prev_participants = prev_frame.get('participantFrames', {})
        game_minute = prev_frame['timestamp'] // 60000

        blue = {'gold': 0, 'level': 0, 'alive': 0, 'cs': 0}
        red  = {'gold': 0, 'level': 0, 'alive': 0, 'cs': 0}

        for pid_str, stats in prev_participants.items():
            pid = int(pid_str)
            bucket = blue if pid <= 5 else red
            bucket['gold']  += stats.get('totalGold', 0)
            bucket['level'] += stats.get('level', 0)
            bucket['cs']    += (stats.get('minionsKilled', 0) +
                                stats.get('jungleMinionsKilled', 0))
            if self._is_participant_alive(pid, frame_idx, frames, game_minute):
                bucket['alive'] += 1

        ally_items, enemy_items = self._count_completed_items(prev_participants)
        recent_blue, recent_red = self._count_recent_kills(frame_idx, frames, 120_000)
        ally_towers, enemy_towers = self._count_towers(frame_idx, frames)
        time_since_fight = self._time_since_last_teamfight(frame_idx, frames)

        elapsed_min = max(1, prev_frame['timestamp'] / 60_000)

        game_state: Dict = {
            # Aggregate stats
            'level_diff':       blue['level'] - red['level'],
            'ally_alive':       blue['alive'],
            'enemy_alive':      red['alive'],
            'game_time_minutes': game_minute,

            # CS rates
            'cs_per_min_diff':  (blue['cs'] / elapsed_min) - (red['cs'] / elapsed_min),
            'ally_cs_per_min':  blue['cs'] / elapsed_min,
            'enemy_cs_per_min': red['cs'] / elapsed_min,

            # Item counts (estimated from gold spent; see _count_completed_items)
            'ally_completed_items':  ally_items,
            'enemy_completed_items': enemy_items,
            'item_advantage':        ally_items - enemy_items,

            # Recent kills (kill feed visible to all players)
            'kills_last_2min_ally':  recent_blue,
            'kills_last_2min_enemy': recent_red,
            'kill_diff_recent':      recent_blue - recent_red,

            # Map control
            'ally_towers':   ally_towers,
            'enemy_towers':  enemy_towers,
            'tower_diff':    ally_towers - enemy_towers,

            # Timing
            'time_since_last_fight': time_since_fight,
        }

        # One-hot encode drake type using dict lookup (replaces long if/elif chain)
        drake_flags = {name: 0 for name in set(DRAKE_KEYWORDS.values())}
        monster_type    = event.get('monsterType', '')
        monster_subtype = event.get('monsterSubType', '')
        drake_type = 'NONE'

        for source in [monster_subtype.upper(), monster_type.upper()]:
            for keyword, canonical in DRAKE_KEYWORDS.items():
                if keyword in source:
                    drake_flags[canonical] = 1
                    drake_type = canonical
                    break
            if drake_type != 'NONE':
                break

        for name, flag in drake_flags.items():
            game_state[f'is_{name.lower()}'] = flag

        game_state['drake_type'] = drake_type  # keep for debugging
        game_state['is_baron']   = 1 if 'BARON' in monster_type.upper() else 0

        return game_state

    def _is_participant_alive(self, participant_id: int,
                              frame_idx: int, frames: List,
                              game_minute: int = 15) -> bool:
        """
        Check if a participant is alive at frame_idx.
        Uses a game-time-scaled respawn window instead of a hardcoded 30s.
        """
        respawn_window_ms = self._estimate_respawn_time_ms(game_minute)
        current_ts = frames[frame_idx].get('timestamp', 0)
        lookback = 2

        for i in range(max(0, frame_idx - lookback), frame_idx):
            for event in frames[i].get('events', []):
                if (event.get('type') == 'CHAMPION_KILL' and
                        event.get('victimId') == participant_id):
                    if current_ts - event.get('timestamp', 0) < respawn_window_ms:
                        return False
        return True

    def _get_killer_team(self, event: Dict, match: Dict) -> int:
        """Return the teamId (100 or 200) of the objective killer."""
        killer_id = event.get('killerId', 0)
        for participant in match['info']['participants']:
            if participant['participantId'] == killer_id:
                return participant['teamId']
        return 100  # default to blue on unknown

    def _count_recent_kills(self, frame_idx: int, frames: List,
                            lookback_ms: int = 120_000) -> Tuple[int, int]:
        """Count champion kills in the last `lookback_ms` ms for each team."""
        blue_kills = red_kills = 0
        current_ts = frames[frame_idx].get('timestamp', 0)

        for i in range(max(0, frame_idx - 10), frame_idx):
            for event in frames[i].get('events', []):
                if event.get('type') != 'CHAMPION_KILL':
                    continue
                if current_ts - event.get('timestamp', 0) > lookback_ms:
                    continue
                killer_id = event.get('killerId', 0)
                if 1 <= killer_id <= 5:
                    blue_kills += 1
                elif 6 <= killer_id <= 10:
                    red_kills += 1

        return blue_kills, red_kills

    def _count_towers(self, frame_idx: int, frames: List) -> Tuple[int, int]:
        """
        Count remaining towers for blue (ally) and red (enemy).
        teamId in BUILDING_KILL is the team that *lost* the tower.
        Starting count of 11 = 3 lanes × 3 tier towers + 2 nexus towers.
        """
        blue_towers = red_towers = 11

        for i in range(frame_idx):
            for event in frames[i].get('events', []):
                if event.get('type') == 'BUILDING_KILL' and 'TOWER' in event.get('buildingType', ''):
                    team_id = event.get('teamId', 0)
                    if team_id == 100:
                        blue_towers -= 1
                    elif team_id == 200:
                        red_towers -= 1

        return max(0, blue_towers), max(0, red_towers)

    def _time_since_last_teamfight(self, frame_idx: int, frames: List) -> int:
        """Return seconds since the last frame with 3+ champion kills. Default 300s."""
        current_ts = frames[frame_idx].get('timestamp', 0)

        for i in range(frame_idx - 1, max(0, frame_idx - 20), -1):
            kills = sum(1 for e in frames[i].get('events', [])
                        if e.get('type') == 'CHAMPION_KILL')
            if kills >= 3:
                return (current_ts - frames[i].get('timestamp', 0)) // 1000

        return 300

    def _count_completed_items(self, participant_frames: Dict) -> Tuple[int, int]:
        """
        Estimate completed items per team from gold spent.
        Uses 3000g per item as a rough average.
        NOTE: this is an approximation — rename if you later use actual item IDs.
        """
        ally_items = enemy_items = 0

        for pid_str, stats in participant_frames.items():
            pid = int(pid_str)
            gold_spent = stats.get('totalGold', 0) - stats.get('currentGold', 0)
            estimated = gold_spent // 3000

            if pid <= 5:
                ally_items += estimated
            else:
                enemy_items += estimated

        return int(ally_items), int(enemy_items)

    # ------------------------------------------------------------------
    # Main collection entrypoint
    # ------------------------------------------------------------------

    def collect_training_data(self, num_matches: int = 100,
                              output_file: str = 'training_data.csv') -> pd.DataFrame:
        """Orchestrate full data collection pipeline and save results."""

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}", flush=True)

        print("Step 1: Getting high-elo players (Grandmaster + Challenger)...", flush=True)
        puuids = self.get_grandmaster_players(count=10)
        for p in self.get_challenger_players(count=10):
            if p not in puuids:
                puuids.append(p)

        if len(puuids) < 5:
            print("WARNING: Very few PUUIDs — check API key and rate limits.", flush=True)
        if not puuids:
            print("ERROR: Could not get any player PUUIDs!")
            return pd.DataFrame()

        print(f"\nStep 2: Getting match IDs from {len(puuids)} players...", flush=True)
        match_ids = self.get_match_ids(puuids, matches_per_player=10)

        if not match_ids:
            print("ERROR: No match IDs found!")
            return pd.DataFrame()

        match_ids = match_ids[:num_matches]

        print(f"\nStep 3: Collecting data from {len(match_ids)} matches...", flush=True)
        all_samples: List[Dict] = []

        for i, match_id in enumerate(match_ids):
            print(f"Processing match {i+1}/{len(match_ids)}: {match_id}", flush=True)
            match_data = self.get_match_data(match_id)
            if match_data:
                samples = self.extract_objective_events(match_data)
                all_samples.extend(samples)
                print(f"  -> {len(samples)} events extracted. "
                      f"Running total: {len(all_samples)}", flush=True)
            else:
                print("  -> Failed to fetch match data.", flush=True)

        if not all_samples:
            print("\nWARNING: No data collected!")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = self._save_with_dedup(df, output_file)
        print(f"\nComplete! {len(df)} total samples saved to {output_file}", flush=True)
        return df


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_env_file = os.path.join(script_dir, 'api.env')

    API_KEY = None

    if os.path.exists(api_env_file):
        try:
            with open(api_env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('RIOT_API_KEY='):
                        API_KEY = line.split('=', 1)[1].strip()
                        break
                    if line.startswith('RGAPI-'):
                        API_KEY = line
                        break
        except Exception as e:
            print(f"Warning: Could not read api.env file: {e}")

    if not API_KEY:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        API_KEY = os.getenv("RIOT_API_KEY")

    if not API_KEY:
        raise ValueError(
            "RIOT_API_KEY not found. Check:\n"
            "  1. api.env file exists with: RIOT_API_KEY=your-key\n"
            "  2. Or set environment variable: export RIOT_API_KEY='your-key'"
        )

    collector = RiotDataCollector(API_KEY)

    output_file = os.path.join(script_dir, 'data', 'objective_training_data.csv')
    df = collector.collect_training_data(num_matches=50, output_file=output_file)

    if len(df) > 0:
        print("\n" + "=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"\nObjective distribution:\n{df['objective_type'].value_counts()}")
        print(f"\nSuccess rate: {df['objective_secured'].mean():.1%}")