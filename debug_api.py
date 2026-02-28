import requests
import json

# Test script to debug Riot API issues


REGION = 'na1'

def test_api_key():
    """Test if API key is valid"""
    print("="*60)
    print("TEST 1: API Key Validation")
    print("="*60)
    
    url = f'https://{REGION}.api.riotgames.com/lol/status/v4/platform-data'
    headers = {'X-Riot-Token': API_KEY}
    
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ API Key is valid!")
        return True
    elif response.status_code == 403:
        print("✗ API Key is invalid or expired")
        print("Go to https://developer.riotgames.com/ to get a new key")
        return False
    elif response.status_code == 401:
        print("✗ Unauthorized - check your API key")
        return False
    else:
        print(f"✗ Unexpected error: {response.text}")
        return False

def test_challenger_endpoint():
    """Test if we can get challenger players"""
    print("\n" + "="*60)
    print("TEST 2: Challenger League Endpoint")
    print("="*60)
    
    url = f'https://{REGION}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
    headers = {'X-Riot-Token': API_KEY}
    
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        entries = data.get('entries', [])
        print(f"✓ Found {len(entries)} challenger players")
        
        if len(entries) > 0:
            print(f"\nFirst entry structure:")
            first_entry = entries[0]
            print(f"Keys: {list(first_entry.keys())}")
            print(f"Sample: {json.dumps(first_entry, indent=2)}")
        
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_summoner_lookup():
    """Test getting a summoner by ID"""
    print("\n" + "="*60)
    print("TEST 3: Get Summoner Info")
    print("="*60)
    
    # First get a challenger player
    url = f'https://{REGION}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
    headers = {'X-Riot-Token': API_KEY}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("✗ Can't get challenger data to test with")
        return False
    
    data = response.json()
    entries = data.get('entries', [])
    
    if len(entries) == 0:
        print("✗ No entries found")
        return False
    
    # Try to get summoner ID from first entry
    first_entry = entries[0]
    summoner_id = first_entry.get('summonerId') or first_entry.get('id')
    
    if not summoner_id:
        print(f"✗ Can't find summonerId in entry: {list(first_entry.keys())}")
        return False
    
    print(f"Testing with summoner ID: {summoner_id}")
    
    # Get summoner details
    summoner_url = f'https://{REGION}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}'
    summ_response = requests.get(summoner_url, headers=headers)
    
    print(f"Status Code: {summ_response.status_code}")
    
    if summ_response.status_code == 200:
        summoner_data = summ_response.json()
        print(f"✓ Got summoner data")
        print(f"PUUID: {summoner_data.get('puuid', 'N/A')[:20]}...")
        return summoner_data.get('puuid')
    else:
        print(f"✗ Error: {summ_response.text}")
        return False

def test_match_history(puuid):
    """Test getting match history for a PUUID"""
    print("\n" + "="*60)
    print("TEST 4: Get Match History")
    print("="*60)
    
    url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids'
    headers = {'X-Riot-Token': API_KEY}
    params = {
        'start': 0,
        'count': 5,
        'queue': 420  # Ranked Solo/Duo
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        match_ids = response.json()
        print(f"✓ Found {len(match_ids)} matches")
        
        if len(match_ids) > 0:
            print(f"Sample match IDs:")
            for match_id in match_ids[:3]:
                print(f"  - {match_id}")
        else:
            print("⚠ No ranked matches found for this player")
        
        return match_ids
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return []

def test_match_details(match_id):
    """Test getting match details"""
    print("\n" + "="*60)
    print("TEST 5: Get Match Details")
    print("="*60)
    
    url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}'
    headers = {'X-Riot-Token': API_KEY}
    
    response = requests.get(url, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ Successfully retrieved match data")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

if __name__ == '__main__':
    print("RIOT API DEBUGGING TOOL")
    print("="*60)
    print(f"Testing with region: {REGION}")
    print(f"API Key (first 10 chars): {API_KEY[:10]}...")
    print()
    
    # Run tests
    if not test_api_key():
        print("\n❌ API key test failed. Fix your API key first.")
        exit(1)
    
    if not test_challenger_endpoint():
        print("\n❌ Can't access challenger data. Check your region.")
        exit(1)
    
    puuid = test_summoner_lookup()
    if not puuid:
        print("\n❌ Can't get PUUID from summoner.")
        exit(1)
    
    match_ids = test_match_history(puuid)
    if len(match_ids) == 0:
        print("\n❌ No matches found. This might be why you're getting 0 matches.")
        print("Try a different region or the player has no recent ranked games.")
        exit(1)
    
    test_match_details(match_ids[0])
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("Your API setup looks good!")
    print("If data collector still fails, it might be a rate limit issue.")