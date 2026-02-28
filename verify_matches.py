"""
Script to verify that collected match data is from real games
"""
import pandas as pd
import requests
import os

def verify_match_via_api(match_id: str, api_key: str) -> dict:
    """Verify a match exists via Riot API"""
    # Extract region from match ID (e.g., NA1_123456 -> na1)
    region = match_id.split('_')[0].lower()
    
    # Use Americas endpoint for match data
    url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}'
    headers = {'X-Riot-Token': api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            match_data = response.json()
            return {
                'exists': True,
                'game_duration': match_data['info']['gameDuration'] // 60,  # minutes
                'game_mode': match_data['info']['gameMode'],
                'queue_id': match_data['info']['queueId'],
                'participants': len(match_data['info']['participants'])
            }
        else:
            return {'exists': False, 'error': f'Status {response.status_code}'}
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def verify_data_quality(df: pd.DataFrame) -> dict:
    """Check if data looks realistic"""
    checks = {}
    
    # Check 1: Match IDs are valid format
    match_ids = df['match_id'].unique()
    valid_regions = ['NA1', 'EUW1', 'EUN1', 'KR', 'BR1', 'LA1', 'LA2', 'OC1', 'RU', 'TR1', 'JP1']
    valid_format = all(any(m.startswith(region + '_') for region in valid_regions) for m in match_ids)
    checks['match_id_format'] = valid_format
    
    # Check 2: Reasonable gold differences (not too extreme)
    max_gold_diff = abs(df['gold_diff']).max()
    checks['reasonable_gold_diff'] = max_gold_diff < 50000  # 50k is very high but possible
    
    # Check 3: Game time is reasonable (5-60 minutes typical)
    game_times = df['game_time_minutes']
    checks['reasonable_game_time'] = game_times.min() >= 0 and game_times.max() <= 90
    
    # Check 4: Players alive is 0-5
    checks['valid_player_counts'] = (
        df['ally_alive'].min() >= 0 and df['ally_alive'].max() <= 5 and
        df['enemy_alive'].min() >= 0 and df['enemy_alive'].max() <= 5
    )
    
    # Check 5: CS values are reasonable
    # Per player: ~350 max by 30 min (very high), ~300 normal
    # Per team (5 players): ~1750 max by 30 min, ~1500 normal
    # Account for game time - longer games = more CS
    max_cs = max(df['ally_total_cs'].max(), df['enemy_total_cs'].max())
    max_game_time = df['game_time_minutes'].max()
    
    # Calculate max acceptable CS based on game time
    # Rough estimate: ~50 CS per minute per team (10 per player per minute)
    max_acceptable_cs = max_game_time * 50 * 1.2  # 20% buffer for very high CS games
    
    checks['reasonable_cs'] = max_cs < max_acceptable_cs
    
    # Also check per-player CS doesn't exceed 350
    df_temp = df.copy()
    df_temp['max_cs_per_player'] = df_temp[['ally_total_cs', 'enemy_total_cs']].max(axis=1) / 5
    checks['reasonable_cs_per_player'] = df_temp['max_cs_per_player'].max() <= 400  # Allow some buffer
    
    # Check 6: No duplicate match_id + timestamp + objective combinations
    duplicates = df.duplicated(subset=['match_id', 'timestamp_minutes', 'objective_type']).sum()
    checks['no_duplicates'] = duplicates == 0
    
    return checks


def main():
    print("="*60)
    print("MATCH DATA VERIFICATION")
    print("="*60)
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'objective_training_data.csv')
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"\nLoaded {len(df)} samples from {df['match_id'].nunique()} unique matches")
    
    # 1. Data Quality Checks
    print("\n" + "="*60)
    print("1. DATA QUALITY CHECKS")
    print("="*60)
    
    checks = verify_data_quality(df)
    all_passed = True
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {check_name:30s}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n   ✓ All data quality checks passed!")
    else:
        print("\n   ⚠️  Some checks failed - review data")
    
    # 2. Sample Statistics
    print("\n" + "="*60)
    print("2. SAMPLE STATISTICS")
    print("="*60)
    
    print(f"\n   Match IDs: {df['match_id'].nunique()} unique")
    print(f"   Regions: {set([m.split('_')[0] for m in df['match_id'].unique()])}")
    print(f"   Game time range: {df['game_time_minutes'].min()}-{df['game_time_minutes'].max()} minutes")
    print(f"   Gold diff range: {df['gold_diff'].min():.0f} to {df['gold_diff'].max():.0f}")
    
    # CS analysis
    max_ally_cs = df['ally_total_cs'].max()
    max_enemy_cs = df['enemy_total_cs'].max()
    max_cs_per_player = max(max_ally_cs, max_enemy_cs) / 5
    
    print(f"   CS range: {df['ally_total_cs'].min()}-{max_ally_cs} (ally), {df['enemy_total_cs'].min()}-{max_enemy_cs} (enemy)")
    print(f"   Max per-player CS: {max_cs_per_player:.0f} (should be <350 for normal games)")
    
    if max_cs_per_player > 350:
        print(f"   ⚠️  WARNING: Some entries have very high CS per player (>350)")
        # Show suspicious entries
        df_temp = df.copy()
        df_temp['ally_cs_per_player'] = df_temp['ally_total_cs'] / 5
        df_temp['enemy_cs_per_player'] = df_temp['enemy_total_cs'] / 5
        suspicious = df_temp[(df_temp['ally_cs_per_player'] > 350) | (df_temp['enemy_cs_per_player'] > 350)]
        if len(suspicious) > 0:
            print(f"   Found {len(suspicious)} entries with high CS:")
            print(suspicious[['match_id', 'game_time_minutes', 'ally_total_cs', 'enemy_total_cs']].head())
    else:
        print(f"   ✓ CS values look reasonable")
    
    # 3. Verify via op.gg links
    print("\n" + "="*60)
    print("3. VERIFY MATCHES ON OP.GG")
    print("="*60)
    
    sample_matches = df['match_id'].unique()[:5]
    print(f"\n   Sample matches to verify:")
    for match_id in sample_matches:
        # Convert NA1_123456 to NA1/123456 for op.gg
        opgg_id = match_id.replace('_', '/')
        print(f"   {match_id}: https://www.op.gg/matches/{opgg_id}")
    
    # 4. Optional: Verify via API (if API key provided)
    print("\n" + "="*60)
    print("4. API VERIFICATION (Optional)")
    print("="*60)
    
    api_key = os.environ.get('RIOT_API_KEY') or 'RGAPI-ffe7f895-7583-48cd-b7fd-7e7f46ebd9aa'
    
    if api_key:
        print(f"\n   Verifying sample match via Riot API...")
        sample_match = df['match_id'].iloc[0]
        result = verify_match_via_api(sample_match, api_key)
        
        if result.get('exists'):
            print(f"   ✓ Match {sample_match} exists!")
            print(f"     Duration: {result['game_duration']} minutes")
            print(f"     Mode: {result['game_mode']}")
            print(f"     Queue: {result['queue_id']}")
            print(f"     Participants: {result['participants']}")
        else:
            print(f"   ✗ Could not verify: {result.get('error', 'Unknown error')}")
    else:
        print("   (Set RIOT_API_KEY environment variable to enable API verification)")
    
    # 5. Duplicate Check
    print("\n" + "="*60)
    print("5. DUPLICATE CHECK")
    print("="*60)
    
    # Check 1: Exact duplicate rows
    exact_dups = len(df) - len(df.drop_duplicates())
    print(f"\n   Exact duplicate rows: {exact_dups}")
    if exact_dups > 0:
        print(f"   ⚠️  Found {exact_dups} exact duplicate rows")
        print(df[df.duplicated(keep=False)].head())
    else:
        print("   ✓ No exact duplicate rows")
    
    # Check 2: Duplicate objectives (same match, time, type)
    obj_dups = df.duplicated(subset=['match_id', 'timestamp_minutes', 'objective_type'], keep=False)
    if obj_dups.any():
        print(f"\n   ⚠️  Found {obj_dups.sum()} duplicate objective events:")
        print(df[obj_dups][['match_id', 'timestamp_minutes', 'objective_type', 'drake_type']].sort_values(['match_id', 'timestamp_minutes']))
    else:
        print("   ✓ No duplicate objectives (same match/time/type)")
    
    # Check 3: Near-duplicates (same match, within 1 minute, same type)
    df_sorted = df.sort_values(['match_id', 'timestamp_minutes'])
    near_dups = []
    for i in range(len(df_sorted) - 1):
        if (df_sorted.iloc[i]['match_id'] == df_sorted.iloc[i+1]['match_id'] and
            abs(df_sorted.iloc[i]['timestamp_minutes'] - df_sorted.iloc[i+1]['timestamp_minutes']) <= 1 and
            df_sorted.iloc[i]['objective_type'] == df_sorted.iloc[i+1]['objective_type']):
            near_dups.append(i)
    
    if near_dups:
        print(f"\n   ⚠️  Found {len(near_dups)} near-duplicates (within 1 minute):")
        for idx in near_dups[:5]:
            row = df_sorted.iloc[idx]
            print(f"     {row['match_id']} at {row['timestamp_minutes']} min ({row['objective_type']})")
    else:
        print("   ✓ No near-duplicates found")
    
    # Check 4: Match frequency (should be reasonable)
    match_counts = df['match_id'].value_counts()
    avg_per_match = match_counts.mean()
    max_per_match = match_counts.max()
    print(f"\n   Match frequency stats:")
    print(f"     Average objectives per match: {avg_per_match:.1f}")
    print(f"     Max objectives in one match: {max_per_match}")
    if max_per_match > 15:
        print(f"     ⚠️  Some matches have unusually many objectives")
        print(f"     Matches with >15 objectives: {len(match_counts[match_counts > 15])}")
    else:
        print("     ✓ Objective counts per match look reasonable")
    
    # 6. Data Distribution
    print("\n" + "="*60)
    print("6. DATA DISTRIBUTION")
    print("="*60)
    
    print(f"\n   Objectives:")
    print(df['objective_type'].value_counts())
    print(f"\n   Drake types:")
    print(df['drake_type'].value_counts())
    print(f"\n   Success rate: {df['objective_secured'].mean():.1%}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print("\nTo verify matches manually:")
    print("1. Visit op.gg links above")
    print("2. Check match IDs match Riot's format")
    print("3. Verify game times and stats make sense")
    print("4. Check that objectives occurred at the times recorded")


if __name__ == '__main__':
    main()

