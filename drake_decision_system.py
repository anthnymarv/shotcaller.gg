import pandas as pd
from typing import List, Dict

class DrakePrioritySystem:
    """
    Weighted decision system for drake priority
    30% personal benefit
    35% team benefit
    35% deny enemy benefit
    """
    
    def __init__(self, champion_data_file='../data/champion_dragon_stats.csv'):
        """Load pre-calculated champion dragon stats"""
        self.champion_stats = pd.read_csv(champion_data_file)
        
        # Create lookup for quick access
        self.benefit_lookup = {}
        for _, row in self.champion_stats.iterrows():
            key = (row['champion'], row['drake_type'], row['num_drakes'])
            self.benefit_lookup[key] = row['value_score']
    
    def get_champion_benefit(self, champion: str, drake_type: str, num_drakes: int = 1) -> float:
        """Get how much a champion benefits from a drake"""
        key = (champion, drake_type, num_drakes)
        return self.benefit_lookup.get(key, 0.0)
    
    def calculate_team_benefit_percentage(self, team_champions: List[str], 
                                         drake_type: str, num_drakes: int = 1) -> float:
        """
        Calculate what percentage of team benefits from this drake
        Returns value between 0-100
        """
        total_benefit = 0
        for champ in team_champions:
            benefit = self.get_champion_benefit(champ, drake_type, num_drakes)
            total_benefit += benefit
        
        # Normalize by number of teammates (out of 9 total players on both teams)
        # If all 5 teammates benefit equally, that's 5/9 = 55.6%
        team_size = len(team_champions)
        return (team_size / 9) * 100
    
    def calculate_drake_priority(self, 
                                your_champion: str,
                                ally_champions: List[str],
                                enemy_champions: List[str],
                                drake_type: str,
                                num_drakes: int = 1,
                                enemy_team_alive: int = 5,
                                drake_alive: bool = True,
                                baron_alive: bool = True) -> Dict:
        """
        Calculate if you should prioritize taking this drake
        
        Returns decision with breakdown
        """
        
        # OVERRIDE: If enemy team is aced and objective is up, ALWAYS take it
        if enemy_team_alive == 0 and (drake_alive or baron_alive):
            return {
                'recommendation': 'TAKE IMMEDIATELY',
                'priority_score': 100,
                'reasoning': 'Enemy team is dead - free objective!',
                'breakdown': {
                    'personal_benefit': 30,
                    'team_benefit': 35,
                    'deny_enemy': 35
                },
                'override': True
            }
        
        # Calculate personal benefit (30%)
        your_benefit = self.get_champion_benefit(your_champion, drake_type, num_drakes)
        
        # Normalize to 0-30 scale (assuming max benefit is around 50)
        personal_score = min((your_benefit / 50) * 30, 30)
        
        # Calculate team benefit (35%)
        # How many of your 4 teammates benefit?
        ally_benefits = []
        for champ in ally_champions:
            benefit = self.get_champion_benefit(champ, drake_type, num_drakes)
            ally_benefits.append(benefit)
        
        # Count how many teammates have significant benefit (score > 5)
        teammates_benefiting = sum(1 for b in ally_benefits if b > 5)
        
        # Calculate percentage: teammates_benefiting out of 9 total players
        team_benefit_percent = (teammates_benefiting / 9) * 100
        team_score = (team_benefit_percent / 100) * 35
        
        # Calculate deny enemy benefit (35%)
        # How many enemies would benefit if THEY got it?
        enemy_benefits = []
        for champ in enemy_champions:
            benefit = self.get_champion_benefit(champ, drake_type, num_drakes)
            enemy_benefits.append(benefit)
        
        # Count enemies who would benefit significantly
        enemies_benefiting = sum(1 for b in enemy_benefits if b > 5)
        
        # Calculate percentage: enemies_benefiting out of 9 total players
        deny_percent = (enemies_benefiting / 9) * 100
        deny_score = (deny_percent / 100) * 35
        
        # Total priority score (0-100)
        total_score = personal_score + team_score + deny_score
        
        # Generate recommendation
        if total_score >= 70:
            recommendation = 'HIGH PRIORITY - Contest this drake'
        elif total_score >= 50:
            recommendation = 'MEDIUM PRIORITY - Take if safe'
        elif total_score >= 30:
            recommendation = 'LOW PRIORITY - Only if free'
        else:
            recommendation = 'SKIP - Not worth contesting'
        
        # Generate detailed reasoning
        reasoning = []
        
        if personal_score > 10:
            reasoning.append(f"You ({your_champion}) benefit significantly from this drake")
        
        if teammates_benefiting >= 3:
            reasoning.append(f"{teammates_benefiting} teammates benefit from {drake_type} drake")
        elif teammates_benefiting == 0:
            reasoning.append(f"Your team doesn't benefit much from {drake_type} drake")
        
        if enemies_benefiting >= 3:
            reasoning.append(f"IMPORTANT: {enemies_benefiting} enemies would benefit - deny them!")
        elif enemies_benefiting <= 1:
            reasoning.append(f"Enemy team doesn't benefit much - less urgent to contest")
        
        return {
            'recommendation': recommendation,
            'priority_score': round(total_score, 1),
            'reasoning': reasoning,
            'breakdown': {
                'personal_benefit': round(personal_score, 1),
                'team_benefit': round(team_score, 1),
                'deny_enemy': round(deny_score, 1)
            },
            'details': {
                'your_champion': your_champion,
                'teammates_benefiting': f"{teammates_benefiting}/4",
                'enemies_benefiting': f"{enemies_benefiting}/5",
                'drake_type': drake_type
            },
            'override': False
        }
    
    def compare_all_drakes(self,
                          your_champion: str,
                          ally_champions: List[str],
                          enemy_champions: List[str]) -> pd.DataFrame:
        """
        Compare priority for all drake types to see which would be best/worst
        """
        results = []
        
        for drake_type in ['INFERNAL', 'MOUNTAIN', 'OCEAN', 'CLOUD', 'HEXTECH', 'CHEMTECH']:
            decision = self.calculate_drake_priority(
                your_champion,
                ally_champions,
                enemy_champions,
                drake_type
            )
            
            results.append({
                'drake_type': drake_type,
                'priority_score': decision['priority_score'],
                'recommendation': decision['recommendation'],
                'personal': decision['breakdown']['personal_benefit'],
                'team': decision['breakdown']['team_benefit'],
                'deny': decision['breakdown']['deny_enemy']
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('priority_score', ascending=False)
        
        return df


def example_usage():
    """Example of how to use the system"""
    
    # Initialize system
    system = DrakePrioritySystem('../data/champion_dragon_stats.csv')
    
    print("="*70)
    print("DRAKE PRIORITY DECISION SYSTEM")
    print("="*70)
    
    # Example game scenario
    your_champion = 'Jinx'
    ally_team = ['Thresh', 'Zed', 'LeeSin', 'Orianna']
    enemy_team = ['Malphite', 'Sejuani', 'Nautilus', 'Ashe', 'Lux']
    drake_type = 'MOUNTAIN'
    
    print(f"\nScenario:")
    print(f"  You: {your_champion}")
    print(f"  Allies: {', '.join(ally_team)}")
    print(f"  Enemies: {', '.join(enemy_team)}")
    print(f"  Drake: {drake_type}")
    
    # Get decision
    decision = system.calculate_drake_priority(
        your_champion=your_champion,
        ally_champions=ally_team,
        enemy_champions=enemy_team,
        drake_type=drake_type,
        num_drakes=1,
        enemy_team_alive=5
    )
    
    print(f"\n{'='*70}")
    print(f"DECISION: {decision['recommendation']}")
    print(f"{'='*70}")
    print(f"Overall Priority Score: {decision['priority_score']}/100")
    
    print(f"\nBreakdown:")
    print(f"  Personal Benefit (30%): {decision['breakdown']['personal_benefit']:.1f}")
    print(f"  Team Benefit (35%):     {decision['breakdown']['team_benefit']:.1f}")
    print(f"  Deny Enemy (35%):       {decision['breakdown']['deny_enemy']:.1f}")
    
    print(f"\nDetails:")
    print(f"  Teammates who benefit: {decision['details']['teammates_benefiting']}")
    print(f"  Enemies who benefit:   {decision['details']['enemies_benefiting']}")
    
    print(f"\nReasoning:")
    for reason in decision['reasoning']:
        print(f"  - {reason}")
    
    # Example 2: Enemy team aced
    print("\n" + "="*70)
    print("SCENARIO 2: ENEMY TEAM ACED")
    print("="*70)
    
    decision_aced = system.calculate_drake_priority(
        your_champion=your_champion,
        ally_champions=ally_team,
        enemy_champions=enemy_team,
        drake_type=drake_type,
        enemy_team_alive=0,  # Enemy team is dead
        drake_alive=True
    )
    
    print(f"\nDECISION: {decision_aced['recommendation']}")
    print(f"Priority Score: {decision_aced['priority_score']}/100")
    print(f"Reasoning: {decision_aced['reasoning']}")
    
    # Example 3: Compare all drakes
    print("\n" + "="*70)
    print("COMPARING ALL DRAKE TYPES")
    print("="*70)
    
    comparison = system.compare_all_drakes(
        your_champion=your_champion,
        ally_champions=ally_team,
        enemy_champions=enemy_team
    )
    
    print(f"\nRanked by priority:")
    print(comparison.to_string(index=False))
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print(f"Best drake for your comp:  {comparison.iloc[0]['drake_type']} "
          f"(score: {comparison.iloc[0]['priority_score']:.1f})")
    print(f"Worst drake for your comp: {comparison.iloc[-1]['drake_type']} "
          f"(score: {comparison.iloc[-1]['priority_score']:.1f})")


if __name__ == '__main__':
    example_usage()
