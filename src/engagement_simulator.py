"""
Community Engagement Simulator
Model the impact of virtual interventions on cohesion indexes
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EngagementSimulator:
    """Simulator for modeling intervention impacts on community cohesion"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.trust_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cohesion_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.sentiment_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.training_data = self._create_training_data()
        self._train_models()
        
    def _create_training_data(self) -> pd.DataFrame:
        """
        Create training data for impact modeling
        In production, this would use real historical data
        """
        np.random.seed(42)
        
        # Generate synthetic training data
        n_samples = 1000
        data = []
        
        for i in range(n_samples):
            # Baseline area characteristics
            baseline = {
                'population': np.random.normal(8000, 2000),
                'unemployment_rate': np.random.beta(2, 8) * 20,
                'crime_rate': np.random.gamma(2, 50),
                'income_deprivation': np.random.beta(2, 8) * 100,
                'education_attainment': np.random.normal(65, 15),
                'ethnic_diversity': np.random.beta(2, 5) * 100,
                'age_dependency_ratio': np.random.normal(0.6, 0.2),
                'housing_stress': np.random.beta(4, 6) * 100,
                'social_trust_baseline': np.random.normal(6.0, 1.5),
                'community_cohesion_baseline': np.random.normal(6.2, 1.4),
                'sentiment_baseline': np.random.normal(6.5, 1.3),
                'volunteer_rate_baseline': np.random.beta(3, 7) * 30
            }
            
            # Intervention investments (0-100 scale)
            interventions = {
                'community_events_investment': np.random.beta(2, 5) * 100,
                'youth_programs_investment': np.random.beta(2, 5) * 100,
                'interfaith_dialogue_investment': np.random.beta(2, 5) * 100,
                'neighborhood_watch_investment': np.random.beta(2, 5) * 100,
                'community_garden_investment': np.random.beta(2, 5) * 100,
                'skills_training_investment': np.random.beta(2, 5) * 100,
                'cultural_exchange_investment': np.random.beta(2, 5) * 100,
                'mental_health_support_investment': np.random.beta(2, 5) * 100,
                'digital_inclusion_investment': np.random.beta(2, 5) * 100,
                'sports_recreation_investment': np.random.beta(2, 5) * 100,
                'volunteer_coordination_investment': np.random.beta(2, 5) * 100,
                'local_business_support_investment': np.random.beta(2, 5) * 100
            }
            
            # Calculate intervention effectiveness based on area characteristics
            effectiveness_modifiers = self._calculate_effectiveness_modifiers(baseline)
            
            # Simulate outcomes
            outcomes = self._simulate_intervention_outcomes(baseline, interventions, effectiveness_modifiers)
            
            # Combine all data
            sample = {**baseline, **interventions, **outcomes}
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _calculate_effectiveness_modifiers(self, baseline: Dict) -> Dict:
        """
        Calculate how area characteristics affect intervention effectiveness
        
        Args:
            baseline: Dictionary with baseline area characteristics
            
        Returns:
            Dictionary with effectiveness modifiers
        """
        modifiers = {}
        
        # Higher deprivation areas may see larger improvements
        deprivation_modifier = baseline['income_deprivation'] / 100
        
        # Areas with lower baseline trust may have more room for improvement
        trust_modifier = max(0, (10 - baseline['social_trust_baseline']) / 10)
        
        # Larger populations may dilute intervention effects
        population_modifier = min(1, baseline['population'] / 10000)
        
        # Higher crime areas may benefit more from safety interventions
        crime_modifier = min(1, baseline['crime_rate'] / 100)
        
        modifiers = {
            'deprivation': deprivation_modifier,
            'trust_potential': trust_modifier,
            'population_size': population_modifier,
            'crime_level': crime_modifier
        }
        
        return modifiers
    
    def _simulate_intervention_outcomes(self, baseline: Dict, interventions: Dict, modifiers: Dict) -> Dict:
        """
        Simulate intervention outcomes based on investments and area characteristics
        
        Args:
            baseline: Baseline area characteristics
            interventions: Investment levels for each intervention
            modifiers: Effectiveness modifiers
            
        Returns:
            Dictionary with outcome metrics
        """
        # Define intervention impact coefficients
        impact_coefficients = {
            'community_events_investment': {'trust': 0.02, 'cohesion': 0.03, 'sentiment': 0.02},
            'youth_programs_investment': {'trust': 0.01, 'cohesion': 0.04, 'sentiment': 0.03},
            'interfaith_dialogue_investment': {'trust': 0.04, 'cohesion': 0.02, 'sentiment': 0.02},
            'neighborhood_watch_investment': {'trust': 0.03, 'cohesion': 0.01, 'sentiment': 0.01},
            'community_garden_investment': {'trust': 0.02, 'cohesion': 0.03, 'sentiment': 0.02},
            'skills_training_investment': {'trust': 0.01, 'cohesion': 0.02, 'sentiment': 0.03},
            'cultural_exchange_investment': {'trust': 0.05, 'cohesion': 0.03, 'sentiment': 0.02},
            'mental_health_support_investment': {'trust': 0.02, 'cohesion': 0.02, 'sentiment': 0.04},
            'digital_inclusion_investment': {'trust': 0.01, 'cohesion': 0.01, 'sentiment': 0.02},
            'sports_recreation_investment': {'trust': 0.02, 'cohesion': 0.04, 'sentiment': 0.03},
            'volunteer_coordination_investment': {'trust': 0.02, 'cohesion': 0.02, 'sentiment': 0.01},
            'local_business_support_investment': {'trust': 0.01, 'cohesion': 0.01, 'sentiment': 0.02}
        }
        
        # Calculate total improvements
        trust_improvement = 0
        cohesion_improvement = 0
        sentiment_improvement = 0
        
        for intervention, investment in interventions.items():
            if intervention in impact_coefficients:
                coeffs = impact_coefficients[intervention]
                
                # Apply investment level (0-100 scale)
                investment_factor = investment / 100
                
                # Apply effectiveness modifiers
                effectiveness = (
                    investment_factor * 
                    (1 + modifiers['deprivation'] * 0.5) * 
                    modifiers['trust_potential'] * 
                    modifiers['population_size']
                )
                
                trust_improvement += coeffs['trust'] * effectiveness
                cohesion_improvement += coeffs['cohesion'] * effectiveness
                sentiment_improvement += coeffs['sentiment'] * effectiveness
        
        # Add some randomness
        noise_factor = np.random.normal(1, 0.1)
        
        # Calculate final scores
        social_trust_final = min(10, baseline['social_trust_baseline'] + trust_improvement * noise_factor)
        community_cohesion_final = min(10, baseline['community_cohesion_baseline'] + cohesion_improvement * noise_factor)
        sentiment_final = min(10, baseline['sentiment_baseline'] + sentiment_improvement * noise_factor)
        
        # Calculate volunteer rate improvement (simplified)
        volunteer_improvement = (trust_improvement + cohesion_improvement) * 2
        volunteer_rate_final = min(50, baseline['volunteer_rate_baseline'] + volunteer_improvement * noise_factor)
        
        return {
            'social_trust_final': social_trust_final,
            'community_cohesion_final': community_cohesion_final,
            'sentiment_final': sentiment_final,
            'volunteer_rate_final': volunteer_rate_final,
            'trust_improvement': social_trust_final - baseline['social_trust_baseline'],
            'cohesion_improvement': community_cohesion_final - baseline['community_cohesion_baseline'],
            'sentiment_improvement': sentiment_final - baseline['sentiment_baseline'],
            'volunteer_improvement': volunteer_rate_final - baseline['volunteer_rate_baseline'],
            'total_investment': sum(interventions.values()),
            'overall_improvement': (
                (social_trust_final - baseline['social_trust_baseline']) * 0.4 +
                (community_cohesion_final - baseline['community_cohesion_baseline']) * 0.4 +
                (sentiment_final - baseline['sentiment_baseline']) * 0.2
            )
        }
    
    def _train_models(self):
        """Train ML models for outcome prediction"""
        # Prepare features
        feature_columns = [
            'population', 'unemployment_rate', 'crime_rate', 'income_deprivation',
            'education_attainment', 'ethnic_diversity', 'age_dependency_ratio',
            'housing_stress', 'social_trust_baseline', 'community_cohesion_baseline',
            'sentiment_baseline', 'volunteer_rate_baseline'
        ]
        
        intervention_columns = [
            'community_events_investment', 'youth_programs_investment',
            'interfaith_dialogue_investment', 'neighborhood_watch_investment',
            'community_garden_investment', 'skills_training_investment',
            'cultural_exchange_investment', 'mental_health_support_investment',
            'digital_inclusion_investment', 'sports_recreation_investment',
            'volunteer_coordination_investment', 'local_business_support_investment'
        ]
        
        X = self.training_data[feature_columns + intervention_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for different outcomes
        self.trust_model.fit(X_scaled, self.training_data['social_trust_final'])
        self.cohesion_model.fit(X_scaled, self.training_data['community_cohesion_final'])
        self.sentiment_model.fit(X_scaled, self.training_data['sentiment_final'])
    
    def simulate_intervention_impact(self, baseline_area: Dict, interventions: Dict) -> Dict:
        """
        Simulate the impact of interventions on a specific area
        
        Args:
            baseline_area: Dictionary with baseline area characteristics
            interventions: Dictionary with intervention investment levels
            
        Returns:
            Dictionary with predicted outcomes
        """
        # Prepare input features
        feature_columns = [
            'population', 'unemployment_rate', 'crime_rate', 'income_deprivation',
            'education_attainment', 'ethnic_diversity', 'age_dependency_ratio',
            'housing_stress', 'social_trust_baseline', 'community_cohesion_baseline',
            'sentiment_baseline', 'volunteer_rate_baseline'
        ]
        
        intervention_columns = [
            'community_events_investment', 'youth_programs_investment',
            'interfaith_dialogue_investment', 'neighborhood_watch_investment',
            'community_garden_investment', 'skills_training_investment',
            'cultural_exchange_investment', 'mental_health_support_investment',
            'digital_inclusion_investment', 'sports_recreation_investment',
            'volunteer_coordination_investment', 'local_business_support_investment'
        ]
        
        # Create input vector
        input_vector = []
        for col in feature_columns:
            input_vector.append(baseline_area.get(col, 0))
        
        for col in intervention_columns:
            input_vector.append(interventions.get(col, 0))
        
        input_vector = np.array(input_vector).reshape(1, -1)
        input_scaled = self.scaler.transform(input_vector)
        
        # Make predictions
        predicted_trust = self.trust_model.predict(input_scaled)[0]
        predicted_cohesion = self.cohesion_model.predict(input_scaled)[0]
        predicted_sentiment = self.sentiment_model.predict(input_scaled)[0]
        
        # Calculate improvements
        trust_improvement = predicted_trust - baseline_area.get('social_trust_baseline', 0)
        cohesion_improvement = predicted_cohesion - baseline_area.get('community_cohesion_baseline', 0)
        sentiment_improvement = predicted_sentiment - baseline_area.get('sentiment_baseline', 0)
        
        # Calculate total investment
        total_investment = sum(interventions.values())
        
        # Calculate cost-effectiveness
        overall_improvement = (
            trust_improvement * 0.4 +
            cohesion_improvement * 0.4 +
            sentiment_improvement * 0.2
        )
        
        cost_effectiveness = overall_improvement / (total_investment / 1000) if total_investment > 0 else 0
        
        return {
            'baseline_area': baseline_area,
            'interventions': interventions,
            'predicted_outcomes': {
                'social_trust_final': round(predicted_trust, 2),
                'community_cohesion_final': round(predicted_cohesion, 2),
                'sentiment_final': round(predicted_sentiment, 2)
            },
            'improvements': {
                'trust_improvement': round(trust_improvement, 2),
                'cohesion_improvement': round(cohesion_improvement, 2),
                'sentiment_improvement': round(sentiment_improvement, 2),
                'overall_improvement': round(overall_improvement, 2)
            },
            'investment_summary': {
                'total_investment': total_investment,
                'cost_effectiveness': round(cost_effectiveness, 3),
                'investment_per_capita': round(total_investment / baseline_area.get('population', 1), 2)
            }
        }
    
    def optimize_intervention_mix(self, baseline_area: Dict, budget: float, 
                                 target_outcome: str = 'overall') -> Dict:
        """
        Optimize intervention mix for given budget and target outcome
        
        Args:
            baseline_area: Dictionary with baseline area characteristics
            budget: Available budget
            target_outcome: Target outcome to optimize ('trust', 'cohesion', 'sentiment', 'overall')
            
        Returns:
            Dictionary with optimized intervention mix
        """
        from scipy.optimize import minimize
        
        intervention_columns = [
            'community_events_investment', 'youth_programs_investment',
            'interfaith_dialogue_investment', 'neighborhood_watch_investment',
            'community_garden_investment', 'skills_training_investment',
            'cultural_exchange_investment', 'mental_health_support_investment',
            'digital_inclusion_investment', 'sports_recreation_investment',
            'volunteer_coordination_investment', 'local_business_support_investment'
        ]
        
        def objective(x):
            # Create interventions dictionary
            interventions = dict(zip(intervention_columns, x))
            
            # Simulate impact
            result = self.simulate_intervention_impact(baseline_area, interventions)
            
            # Return negative of target outcome (minimize negative = maximize positive)
            if target_outcome == 'trust':
                return -result['improvements']['trust_improvement']
            elif target_outcome == 'cohesion':
                return -result['improvements']['cohesion_improvement']
            elif target_outcome == 'sentiment':
                return -result['improvements']['sentiment_improvement']
            else:  # overall
                return -result['improvements']['overall_improvement']
        
        # Constraints: sum of investments <= budget, each investment >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget - sum(x)},
            {'type': 'ineq', 'fun': lambda x: x}  # x >= 0
        ]
        
        # Initial guess: equal distribution
        x0 = [budget / len(intervention_columns)] * len(intervention_columns)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        if result.success:
            optimized_interventions = dict(zip(intervention_columns, result.x))
            simulation_result = self.simulate_intervention_impact(baseline_area, optimized_interventions)
            
            return {
                'optimization_successful': True,
                'optimized_interventions': optimized_interventions,
                'simulation_result': simulation_result,
                'optimization_details': {
                    'target_outcome': target_outcome,
                    'budget': budget,
                    'optimization_method': 'SLSQP',
                    'iterations': result.nit
                }
            }
        else:
            return {
                'optimization_successful': False,
                'error': result.message,
                'fallback_interventions': dict(zip(intervention_columns, x0))
            }
    
    def create_scenario_comparison(self, baseline_area: Dict, scenarios: List[Dict]) -> Dict:
        """
        Compare multiple intervention scenarios
        
        Args:
            baseline_area: Dictionary with baseline area characteristics
            scenarios: List of scenario dictionaries with intervention mixes
            
        Returns:
            Dictionary with scenario comparison results
        """
        comparison_results = []
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario {i+1}')
            interventions = scenario.get('interventions', {})
            
            result = self.simulate_intervention_impact(baseline_area, interventions)
            
            comparison_results.append({
                'scenario_name': scenario_name,
                'interventions': interventions,
                'outcomes': result['predicted_outcomes'],
                'improvements': result['improvements'],
                'investment_summary': result['investment_summary']
            })
        
        # Find best scenario for each outcome
        best_scenarios = {
            'best_trust_improvement': max(comparison_results, key=lambda x: x['improvements']['trust_improvement']),
            'best_cohesion_improvement': max(comparison_results, key=lambda x: x['improvements']['cohesion_improvement']),
            'best_sentiment_improvement': max(comparison_results, key=lambda x: x['improvements']['sentiment_improvement']),
            'best_overall_improvement': max(comparison_results, key=lambda x: x['improvements']['overall_improvement']),
            'best_cost_effectiveness': max(comparison_results, key=lambda x: x['investment_summary']['cost_effectiveness'])
        }
        
        return {
            'baseline_area': baseline_area,
            'scenarios': comparison_results,
            'best_scenarios': best_scenarios,
            'summary': {
                'total_scenarios': len(scenarios),
                'average_trust_improvement': np.mean([s['improvements']['trust_improvement'] for s in comparison_results]),
                'average_cohesion_improvement': np.mean([s['improvements']['cohesion_improvement'] for s in comparison_results]),
                'average_sentiment_improvement': np.mean([s['improvements']['sentiment_improvement'] for s in comparison_results]),
                'average_overall_improvement': np.mean([s['improvements']['overall_improvement'] for s in comparison_results])
            }
        }
    
    def run_full_analysis(self, baseline_area: Optional[Dict] = None) -> Dict:
        """
        Run complete engagement simulation analysis
        
        Args:
            baseline_area: Optional baseline area characteristics
            
        Returns:
            Dictionary with complete analysis results
        """
        if baseline_area is None:
            # Create sample baseline area
            baseline_area = {
                'population': 8500,
                'unemployment_rate': 8.5,
                'crime_rate': 75,
                'income_deprivation': 25,
                'education_attainment': 68,
                'ethnic_diversity': 35,
                'age_dependency_ratio': 0.65,
                'housing_stress': 40,
                'social_trust_baseline': 5.8,
                'community_cohesion_baseline': 6.1,
                'sentiment_baseline': 6.5,
                'volunteer_rate_baseline': 15
            }
        
        # Create sample scenarios
        scenarios = [
            {
                'name': 'Community Focus',
                'interventions': {
                    'community_events_investment': 80,
                    'community_garden_investment': 60,
                    'cultural_exchange_investment': 70,
                    'sports_recreation_investment': 50
                }
            },
            {
                'name': 'Youth & Skills Focus',
                'interventions': {
                    'youth_programs_investment': 90,
                    'skills_training_investment': 80,
                    'digital_inclusion_investment': 60,
                    'sports_recreation_investment': 40
                }
            },
            {
                'name': 'Safety & Trust Focus',
                'interventions': {
                    'neighborhood_watch_investment': 70,
                    'interfaith_dialogue_investment': 80,
                    'mental_health_support_investment': 60,
                    'volunteer_coordination_investment': 50
                }
            }
        ]
        
        # Run scenario comparison
        scenario_comparison = self.create_scenario_comparison(baseline_area, scenarios)
        
        # Optimize for different outcomes
        optimization_results = {}
        for outcome in ['trust', 'cohesion', 'sentiment', 'overall']:
            optimization_results[outcome] = self.optimize_intervention_mix(
                baseline_area, budget=1000, target_outcome=outcome
            )
        
        return {
            'baseline_area': baseline_area,
            'scenario_comparison': scenario_comparison,
            'optimization_results': optimization_results,
            'model_performance': {
                'training_samples': len(self.training_data),
                'trust_model_score': self.trust_model.score(
                    self.scaler.transform(self.training_data.iloc[:, :24]), 
                    self.training_data['social_trust_final']
                ),
                'cohesion_model_score': self.cohesion_model.score(
                    self.scaler.transform(self.training_data.iloc[:, :24]), 
                    self.training_data['community_cohesion_final']
                ),
                'sentiment_model_score': self.sentiment_model.score(
                    self.scaler.transform(self.training_data.iloc[:, :24]), 
                    self.training_data['sentiment_final']
                )
            },
            'analysis_timestamp': pd.Timestamp.now()
        }
