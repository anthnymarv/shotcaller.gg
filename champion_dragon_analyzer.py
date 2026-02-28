import requests
import json
import pandas as pd
from typing import Dict, List

class ChampionDragonAnalyzer:
    """
    Calculates how much each dragon type benefits each champion
    based on their stats and playstyle
    """
    
    def __init__(self):
        self.champions = {}
        self.dragon_buffs = {
            'INFERNAL': {'ad_percent': 4, 'ap_percent': 4},
            'MOUNTAIN': {'armor_percent': 6, 'mr_percent': 6},
            'OCEAN': {'hp_regen_percent': 2.5, 'mana_regen_percent': 2.5},
            'CLOUD': {'ms_percent': 3.5, 'ability_haste': 0},
            'HEXTECH': {'attack_speed_percent': 4, 'ability_haste': 5},
            'CHEMTECH': {'damage_increase_percent': 5}
        }
    
    def fetch_champion_data(self):
        """Fetch champion stats from Data Dragon"""
        print("Fetching champion data from Data Dragon...")
        
        # Get latest version
        version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        try:
            response = requests.get(version_url)
            response.raise_for_status()  # Raise an error for bad status codes
            versions = response.json()
            latest_version = versions[0]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching version data: {e}")
            raise
        
        print(f"Latest patch: {latest_version}")
        
        # Get all champions
        champions_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
        try:
            response = requests.get(champions_url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching champions data: {e}")
            raise
        
        # Get detailed stats for each champion
        for champ_name, champ_info in data['data'].items():
            champ_id = champ_info['id']
            
            # Get detailed champion data
            detail_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion/{champ_id}.json"
            try:
                detail_response = requests.get(detail_url)
                detail_response.raise_for_status()
                detail_data = detail_response.json()
            except requests.exceptions.RequestException as e:
                print(f"Warning: Error fetching data for {champ_name}: {e}")
                continue
            
            stats = detail_data['data'][champ_id]['stats']
            
            self.champions[champ_name] = {
                'name': champ_name,
                'id': champ_id,
                'stats': stats,
                'tags': champ_info['tags']  # Fighter, Mage, Tank, etc.
            }
        
        print(f"Loaded {len(self.champions)} champions")
        return self.champions
    
    def calculate_dragon_value(self, champion_name: str, level: int = 11, 
                               num_drakes: int = 1, drake_type: str = 'INFERNAL') -> Dict:
        """
        Calculate how much a dragon improves a champion's stats
        """
        if champion_name not in self.champions:
            return None
        
        champ = self.champions[champion_name]
        base_stats = champ['stats']
        
        # Calculate stats at given level
        # Formula: base + (growth * (level - 1) * (0.7025 + 0.0175 * (level - 1)))
        def stat_at_level(base, growth, lvl):
            return base + (growth * (lvl - 1) * (0.7025 + 0.0175 * (lvl - 1)))
        
        hp = stat_at_level(base_stats['hp'], base_stats['hpperlevel'], level)
        armor = stat_at_level(base_stats['armor'], base_stats['armorperlevel'], level)
        mr = stat_at_level(base_stats['spellblock'], base_stats['spellblockperlevel'], level)
        ad = stat_at_level(base_stats['attackdamage'], base_stats['attackdamageperlevel'], level)
        
        result = {
            'champion': champion_name,
            'level': level,
            'drake_type': drake_type,
            'num_drakes': num_drakes,
            'base_stats': {
                'hp': round(hp, 1),
                'armor': round(armor, 1),
                'mr': round(mr, 1),
                'ad': round(ad, 1),
                'attack_speed': round(base_stats['attackspeed'], 3)
            },
            'drake_buffs': {},
            'total_value_score': 0
        }
        
        buff = self.dragon_buffs[drake_type]
        multiplier = num_drakes  # Each drake stacks
        
        if drake_type == 'INFERNAL':
            ad_increase = ad * (buff['ad_percent'] / 100) * multiplier
            result['drake_buffs']['ad_increase'] = round(ad_increase, 1)
            result['drake_buffs']['new_ad'] = round(ad + ad_increase, 1)
            
            # AP champions would get AP increase (assume 0 base AP, use items)
            # For simplicity, score AD champions higher
            if 'Marksman' in champ['tags'] or 'Assassin' in champ['tags']:
                result['total_value_score'] = ad_increase * 3
            else:
                result['total_value_score'] = ad_increase
        
        elif drake_type == 'MOUNTAIN':
            armor_increase = armor * (buff['armor_percent'] / 100) * multiplier
            mr_increase = mr * (buff['mr_percent'] / 100) * multiplier
            
            result['drake_buffs']['armor_increase'] = round(armor_increase, 1)
            result['drake_buffs']['mr_increase'] = round(mr_increase, 1)
            result['drake_buffs']['new_armor'] = round(armor + armor_increase, 1)
            result['drake_buffs']['new_mr'] = round(mr + mr_increase, 1)
            
            # Tanks benefit most
            if 'Tank' in champ['tags'] or 'Fighter' in champ['tags']:
                result['total_value_score'] = (armor_increase + mr_increase) * 2
            else:
                result['total_value_score'] = (armor_increase + mr_increase) * 0.5
        
        elif drake_type == 'OCEAN':
            hp_regen = hp * (buff['hp_regen_percent'] / 100) * multiplier
            result['drake_buffs']['hp_regen_per_5s'] = round(hp_regen, 1)
            
            # Benefits everyone but especially sustain/poke comps
            result['total_value_score'] = hp_regen * 1.5
        
        elif drake_type == 'CLOUD':
            ms_increase = base_stats['movespeed'] * (buff['ms_percent'] / 100) * multiplier
            result['drake_buffs']['ms_increase'] = round(ms_increase, 1)
            result['drake_buffs']['new_ms'] = round(base_stats['movespeed'] + ms_increase, 1)
            
            # Benefits engage/kite champions
            if 'Support' in champ['tags'] or 'Mage' in champ['tags']:
                result['total_value_score'] = ms_increase * 2
            else:
                result['total_value_score'] = ms_increase
        
        elif drake_type == 'HEXTECH':
            as_increase = base_stats['attackspeed'] * (buff['attack_speed_percent'] / 100) * multiplier
            result['drake_buffs']['as_increase'] = round(as_increase, 3)
            result['drake_buffs']['new_as'] = round(base_stats['attackspeed'] + as_increase, 3)
            result['drake_buffs']['ability_haste'] = buff['ability_haste'] * multiplier
            
            # Benefits ADCs and auto-attack based champs
            if 'Marksman' in champ['tags']:
                result['total_value_score'] = as_increase * 100 + buff['ability_haste'] * 2
            else:
                result['total_value_score'] = buff['ability_haste'] * 2
        
        elif drake_type == 'CHEMTECH':
            damage_increase = buff['damage_increase_percent'] * multiplier
            result['drake_buffs']['damage_increase_percent'] = damage_increase
            
            # Benefits everyone
            result['total_value_score'] = damage_increase * 5
        
        return result
    
    def rank_champions_for_drake(self, drake_type: str, level: int = 11, 
                                 num_drakes: int = 1, top_n: int = 10) -> pd.DataFrame:
        """
        Rank all champions by how much they benefit from a specific drake
        """
        results = []
        
        for champ_name in self.champions.keys():
            analysis = self.calculate_dragon_value(champ_name, level, num_drakes, drake_type)
            if analysis:
                results.append({
                    'champion': champ_name,
                    'value_score': analysis['total_value_score'],
                    'buffs': analysis['drake_buffs']
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('value_score', ascending=False)
        
        return df.head(top_n)
    
    def analyze_team_composition(self, team_champions: List[str], 
                                 drake_type: str, level: int = 11) -> Dict:
        """
        Analyze how much a specific drake benefits your entire team
        """
        team_value = 0
        champion_details = []
        
        for champ in team_champions:
            analysis = self.calculate_dragon_value(champ, level, 1, drake_type)
            if analysis:
                team_value += analysis['total_value_score']
                champion_details.append({
                    'champion': champ,
                    'value': analysis['total_value_score'],
                    'buffs': analysis['drake_buffs']
                })
        
        return {
            'team': team_champions,
            'drake_type': drake_type,
            'total_team_value': round(team_value, 1),
            'avg_value_per_champion': round(team_value / len(team_champions), 1),
            'champion_breakdown': champion_details,
            'priority': 'HIGH' if team_value > 100 else 'MEDIUM' if team_value > 50 else 'LOW'
        }
    
    def export_all_data_to_csv(self, output_file: str = 'champion_dragon_data.csv'):
        """
        Export complete analysis for all champions and all drakes to CSV
        """
        all_data = []
        
        for champ_name in self.champions.keys():
            for drake_type in self.dragon_buffs.keys():
                for num_drakes in [1, 2, 3, 4]:
                    analysis = self.calculate_dragon_value(
                        champ_name, level=11, num_drakes=num_drakes, drake_type=drake_type
                    )
                    
                    if analysis:
                        row = {
                            'champion': champ_name,
                            'drake_type': drake_type,
                            'num_drakes': num_drakes,
                            'value_score': analysis['total_value_score'],
                            'base_ad': analysis['base_stats'].get('ad', 0),
                            'base_armor': analysis['base_stats'].get('armor', 0),
                            'base_mr': analysis['base_stats'].get('mr', 0),
                        }
                        
                        # Add drake-specific buffs
                        for buff_key, buff_value in analysis['drake_buffs'].items():
                            row[buff_key] = buff_value
                        
                        all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} rows to {output_file}")
        
        return df


if __name__ == '__main__':
    analyzer = ChampionDragonAnalyzer()
    
    # Fetch champion data
    analyzer.fetch_champion_data()
    
    print("\n" + "="*60)
    print("INFERNAL DRAKE - TOP 10 BENEFICIARIES")
    print("="*60)
    top_infernal = analyzer.rank_champions_for_drake('INFERNAL', level=11, num_drakes=1)
    print(top_infernal[['champion', 'value_score']].to_string(index=False))
    
    print("\n" + "="*60)
    print("MOUNTAIN DRAKE - TOP 10 BENEFICIARIES")
    print("="*60)
    top_mountain = analyzer.rank_champions_for_drake('MOUNTAIN', level=11, num_drakes=1)
    print(top_mountain[['champion', 'value_score']].to_string(index=False))
    
    print("\n" + "="*60)
    print("EXAMPLE: TEAM COMPOSITION ANALYSIS")
    print("="*60)
    team = ['Jinx', 'Thresh', 'Zed', 'LeeSin', 'Orianna']
    
    for drake in ['INFERNAL', 'MOUNTAIN', 'OCEAN']:
        analysis = analyzer.analyze_team_composition(team, drake, level=11)
        print(f"\n{drake} Drake:")
        print(f"  Total Team Value: {analysis['total_team_value']}")
        print(f"  Priority: {analysis['priority']}")
        print(f"  Top Beneficiary: {analysis['champion_breakdown'][0]['champion']} "
              f"(+{analysis['champion_breakdown'][0]['value']:.1f} value)")
    
    print("\n" + "="*60)
    print("EXPORTING COMPLETE DATASET")
    print("="*60)
    df = analyzer.export_all_data_to_csv('data/champion_dragon_stats.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")