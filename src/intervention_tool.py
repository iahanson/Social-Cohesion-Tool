"""
Intervention Effectiveness Tool
Case-based recommendation system for social cohesion interventions
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class InterventionTool:
    """Tool for recommending interventions based on similar historical cases"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.intervention_database = self._create_intervention_database()
        
    def _create_intervention_database(self) -> pd.DataFrame:
        """
        Create database of historical interventions and outcomes
        In production, this would be populated from real data sources
        """
        np.random.seed(42)
        
        # Define intervention types
        intervention_types = [
            'Community Events Program',
            'Youth Engagement Initiative',
            'Interfaith Dialogue Program',
            'Neighborhood Watch Enhancement',
            'Community Garden Project',
            'Skills Training Program',
            'Cultural Exchange Program',
            'Mental Health Support Group',
            'Digital Inclusion Program',
            'Sports and Recreation Program',
            'Volunteer Coordination Hub',
            'Local Business Support',
            'Housing Support Services',
            'Education Outreach Program',
            'Environmental Clean-up Initiative'
        ]
        
        # Generate historical cases
        n_cases = 200
        cases = []
        
        for i in range(n_cases):
            # Random area characteristics (pre-intervention)
            case = {
                'case_id': f'CASE_{i+1:04d}',
                'area_type': np.random.choice(['Urban', 'Suburban', 'Rural']),
                'population': np.random.normal(8000, 2000),
                'unemployment_rate': np.random.beta(2, 8) * 20,
                'crime_rate': np.random.gamma(2, 50),
                'social_trust_pre': np.random.normal(6.0, 1.5),
                'community_cohesion_pre': np.random.normal(6.2, 1.4),
                'volunteer_rate_pre': np.random.beta(3, 7) * 30,
                'education_attainment': np.random.normal(65, 15),
                'income_deprivation': np.random.beta(2, 8) * 100,
                'ethnic_diversity': np.random.beta(2, 5) * 100,
                'age_dependency_ratio': np.random.normal(0.6, 0.2),
                'housing_stress': np.random.beta(4, 6) * 100,
                'local_authority_type': np.random.choice(['Metropolitan', 'Unitary', 'District']),
                'region': np.random.choice(['London', 'South East', 'North West', 'Yorkshire', 'West Midlands'])
            }
            
            # Select intervention
            intervention = np.random.choice(intervention_types)
            case['intervention_type'] = intervention
            
            # Intervention characteristics
            case['intervention_duration_months'] = np.random.choice([6, 12, 18, 24, 36])
            case['funding_amount'] = np.random.lognormal(10, 1)  # Log-normal distribution
            case['staff_involvement'] = np.random.choice(['Low', 'Medium', 'High'])
            case['community_participation'] = np.random.choice(['Low', 'Medium', 'High'])
            case['partnership_involvement'] = np.random.choice(['None', 'Local', 'Regional', 'National'])
            
            # Simulate outcomes based on intervention type and area characteristics
            outcomes = self._simulate_outcomes(case, intervention)
            case.update(outcomes)
            
            cases.append(case)
        
        return pd.DataFrame(cases)
    
    def _simulate_outcomes(self, case: Dict, intervention: str) -> Dict:
        """
        Simulate intervention outcomes based on area characteristics and intervention type
        
        Args:
            case: Dictionary with area characteristics
            intervention: Intervention type
            
        Returns:
            Dictionary with outcome metrics
        """
        # Base effectiveness by intervention type
        intervention_effectiveness = {
            'Community Events Program': {'trust': 0.3, 'cohesion': 0.4, 'volunteer': 0.2},
            'Youth Engagement Initiative': {'trust': 0.2, 'cohesion': 0.5, 'volunteer': 0.1},
            'Interfaith Dialogue Program': {'trust': 0.4, 'cohesion': 0.3, 'volunteer': 0.1},
            'Neighborhood Watch Enhancement': {'trust': 0.3, 'cohesion': 0.2, 'volunteer': 0.3},
            'Community Garden Project': {'trust': 0.2, 'cohesion': 0.4, 'volunteer': 0.4},
            'Skills Training Program': {'trust': 0.1, 'cohesion': 0.2, 'volunteer': 0.1},
            'Cultural Exchange Program': {'trust': 0.5, 'cohesion': 0.4, 'volunteer': 0.1},
            'Mental Health Support Group': {'trust': 0.2, 'cohesion': 0.3, 'volunteer': 0.2},
            'Digital Inclusion Program': {'trust': 0.1, 'cohesion': 0.2, 'volunteer': 0.1},
            'Sports and Recreation Program': {'trust': 0.3, 'cohesion': 0.5, 'volunteer': 0.2},
            'Volunteer Coordination Hub': {'trust': 0.2, 'cohesion': 0.3, 'volunteer': 0.6},
            'Local Business Support': {'trust': 0.1, 'cohesion': 0.2, 'volunteer': 0.1},
            'Housing Support Services': {'trust': 0.2, 'cohesion': 0.2, 'volunteer': 0.1},
            'Education Outreach Program': {'trust': 0.2, 'cohesion': 0.3, 'volunteer': 0.2},
            'Environmental Clean-up Initiative': {'trust': 0.3, 'cohesion': 0.4, 'volunteer': 0.3}
        }
        
        base_effects = intervention_effectiveness[intervention]
        
        # Modify effects based on area characteristics
        # Higher deprivation areas may see larger improvements
        deprivation_modifier = case['income_deprivation'] / 100 * 0.5
        
        # Higher initial trust may limit improvement potential
        trust_modifier = max(0, (10 - case['social_trust_pre']) / 10)
        
        # Duration effect (longer interventions generally more effective)
        duration_modifier = min(1.5, case['intervention_duration_months'] / 12)
        
        # Calculate post-intervention scores
        trust_improvement = base_effects['trust'] * (1 + deprivation_modifier) * trust_modifier * duration_modifier
        cohesion_improvement = base_effects['cohesion'] * (1 + deprivation_modifier) * duration_modifier
        volunteer_improvement = base_effects['volunteer'] * duration_modifier
        
        # Add some randomness
        noise_factor = np.random.normal(1, 0.2)
        
        case['social_trust_post'] = min(10, case['social_trust_pre'] + trust_improvement * noise_factor)
        case['community_cohesion_post'] = min(10, case['community_cohesion_pre'] + cohesion_improvement * noise_factor)
        case['volunteer_rate_post'] = min(50, case['volunteer_rate_pre'] + volunteer_improvement * 10 * noise_factor)
        
        # Calculate improvement metrics
        case['trust_improvement'] = case['social_trust_post'] - case['social_trust_pre']
        case['cohesion_improvement'] = case['community_cohesion_post'] - case['community_cohesion_pre']
        case['volunteer_improvement'] = case['volunteer_rate_post'] - case['volunteer_rate_pre']
        
        # Overall success score
        case['success_score'] = (
            case['trust_improvement'] * 0.4 +
            case['cohesion_improvement'] * 0.4 +
            case['volunteer_improvement'] / 10 * 0.2
        )
        
        # Cost-effectiveness (improvement per £1000)
        case['cost_effectiveness'] = case['success_score'] / (case['funding_amount'] / 1000)
        
        return case
    
    def find_similar_cases(self, target_area: Dict, n_similar: int = 10) -> pd.DataFrame:
        """
        Find cases similar to the target area
        
        Args:
            target_area: Dictionary with target area characteristics
            n_similar: Number of similar cases to return
            
        Returns:
            DataFrame with similar cases
        """
        # Prepare features for similarity matching
        feature_columns = [
            'population', 'unemployment_rate', 'crime_rate', 'social_trust_pre',
            'community_cohesion_pre', 'volunteer_rate_pre', 'education_attainment',
            'income_deprivation', 'ethnic_diversity', 'age_dependency_ratio',
            'housing_stress'
        ]
        
        # Create target vector
        target_vector = np.array([target_area.get(col, 0) for col in feature_columns]).reshape(1, -1)
        
        # Prepare database features
        db_features = self.intervention_database[feature_columns].fillna(0)
        
        # Scale features
        db_features_scaled = self.scaler.fit_transform(db_features)
        target_scaled = self.scaler.transform(target_vector)
        
        # Find similar cases
        similarities = cosine_similarity(target_scaled, db_features_scaled)[0]
        
        # Get top similar cases
        similar_indices = np.argsort(similarities)[::-1][:n_similar]
        
        similar_cases = self.intervention_database.iloc[similar_indices].copy()
        similar_cases['similarity_score'] = similarities[similar_indices]
        
        return similar_cases
    
    def recommend_interventions(self, target_area: Dict, n_recommendations: int = 5) -> List[Dict]:
        """
        Recommend interventions for a target area
        
        Args:
            target_area: Dictionary with target area characteristics
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        # Find similar cases
        similar_cases = self.find_similar_cases(target_area, n_similar=50)
        
        # Group by intervention type and calculate average outcomes
        intervention_stats = similar_cases.groupby('intervention_type').agg({
            'success_score': ['mean', 'std', 'count'],
            'cost_effectiveness': ['mean', 'std'],
            'trust_improvement': 'mean',
            'cohesion_improvement': 'mean',
            'volunteer_improvement': 'mean',
            'funding_amount': 'mean',
            'intervention_duration_months': 'mean'
        }).round(3)
        
        # Flatten column names
        intervention_stats.columns = ['_'.join(col).strip() for col in intervention_stats.columns]
        intervention_stats = intervention_stats.reset_index()
        
        # Filter interventions with sufficient data
        intervention_stats = intervention_stats[intervention_stats['success_score_count'] >= 3]
        
        # Calculate recommendation score
        intervention_stats['recommendation_score'] = (
            intervention_stats['success_score_mean'] * 0.4 +
            intervention_stats['cost_effectiveness_mean'] * 0.3 +
            (1 - intervention_stats['success_score_std']) * 0.3  # Prefer consistent results
        )
        
        # Sort by recommendation score
        intervention_stats = intervention_stats.sort_values('recommendation_score', ascending=False)
        
        # Create recommendations
        recommendations = []
        for _, row in intervention_stats.head(n_recommendations).iterrows():
            recommendation = {
                'intervention_type': row['intervention_type'],
                'expected_success_score': row['success_score_mean'],
                'expected_trust_improvement': row['trust_improvement_mean'],
                'expected_cohesion_improvement': row['cohesion_improvement_mean'],
                'expected_volunteer_improvement': row['volunteer_improvement_mean'],
                'average_funding_required': row['funding_amount_mean'],
                'average_duration_months': row['intervention_duration_months_mean'],
                'cost_effectiveness': row['cost_effectiveness_mean'],
                'confidence_level': min(1.0, row['success_score_count'] / 10),  # Based on sample size
                'similar_cases_count': int(row['success_score_count']),
                'evidence_base': f"Based on {int(row['success_score_count'])} similar cases"
            }
            
            # Add specific recommendations based on intervention type
            recommendation['key_components'] = self._get_intervention_components(row['intervention_type'])
            recommendation['success_factors'] = self._get_success_factors(row['intervention_type'])
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_intervention_components(self, intervention_type: str) -> List[str]:
        """Get key components for an intervention type"""
        components = {
            'Community Events Program': [
                'Regular community gatherings',
                'Cultural celebrations',
                'Local market events',
                'Community feedback sessions'
            ],
            'Youth Engagement Initiative': [
                'Youth councils',
                'Skills workshops',
                'Mentorship programs',
                'Recreational activities'
            ],
            'Interfaith Dialogue Program': [
                'Interfaith meetings',
                'Cultural exchange events',
                'Shared community projects',
                'Educational workshops'
            ],
            'Neighborhood Watch Enhancement': [
                'Community safety training',
                'Regular patrols',
                'Communication networks',
                'Crime prevention education'
            ],
            'Community Garden Project': [
                'Shared growing spaces',
                'Gardening workshops',
                'Community harvest events',
                'Environmental education'
            ]
        }
        
        return components.get(intervention_type, [
            'Community engagement',
            'Regular activities',
            'Local partnerships',
            'Ongoing evaluation'
        ])
    
    def _get_success_factors(self, intervention_type: str) -> List[str]:
        """Get success factors for an intervention type"""
        factors = {
            'Community Events Program': [
                'Strong local leadership',
                'Adequate funding',
                'Community participation',
                'Regular scheduling'
            ],
            'Youth Engagement Initiative': [
                'Youth involvement in planning',
                'Skilled facilitators',
                'Safe spaces',
                'Long-term commitment'
            ],
            'Interfaith Dialogue Program': [
                'Respectful dialogue',
                'Diverse participation',
                'Shared goals',
                'Cultural sensitivity'
            ],
            'Neighborhood Watch Enhancement': [
                'Police partnership',
                'Community training',
                'Clear communication',
                'Regular meetings'
            ],
            'Community Garden Project': [
                'Land access',
                'Gardening expertise',
                'Community ownership',
                'Sustainable practices'
            ]
        }
        
        return factors.get(intervention_type, [
            'Community buy-in',
            'Adequate resources',
            'Skilled leadership',
            'Long-term vision'
        ])
    
    def analyze_intervention_effectiveness(self, intervention_type: str) -> Dict:
        """
        Analyze effectiveness of a specific intervention type
        
        Args:
            intervention_type: Type of intervention to analyze
            
        Returns:
            Dictionary with effectiveness analysis
        """
        intervention_cases = self.intervention_database[
            self.intervention_database['intervention_type'] == intervention_type
        ]
        
        if intervention_cases.empty:
            return {'error': 'No cases found for this intervention type'}
        
        analysis = {
            'intervention_type': intervention_type,
            'total_cases': len(intervention_cases),
            'average_success_score': intervention_cases['success_score'].mean(),
            'success_rate': (intervention_cases['success_score'] > 0).mean(),
            'average_funding': intervention_cases['funding_amount'].mean(),
            'average_duration': intervention_cases['intervention_duration_months'].mean(),
            'cost_effectiveness': intervention_cases['cost_effectiveness'].mean(),
            
            'outcomes': {
                'trust_improvement': {
                    'mean': intervention_cases['trust_improvement'].mean(),
                    'std': intervention_cases['trust_improvement'].std(),
                    'positive_rate': (intervention_cases['trust_improvement'] > 0).mean()
                },
                'cohesion_improvement': {
                    'mean': intervention_cases['cohesion_improvement'].mean(),
                    'std': intervention_cases['cohesion_improvement'].std(),
                    'positive_rate': (intervention_cases['cohesion_improvement'] > 0).mean()
                },
                'volunteer_improvement': {
                    'mean': intervention_cases['volunteer_improvement'].mean(),
                    'std': intervention_cases['volunteer_improvement'].std(),
                    'positive_rate': (intervention_cases['volunteer_improvement'] > 0).mean()
                }
            },
            
            'context_factors': {
                'most_effective_in': intervention_cases.groupby('area_type')['success_score'].mean().idxmax(),
                'optimal_duration': intervention_cases.groupby('intervention_duration_months')['success_score'].mean().idxmax(),
                'funding_sweet_spot': intervention_cases.groupby(pd.cut(intervention_cases['funding_amount'], 5))['success_score'].mean().idxmax()
            }
        }
        
        return analysis
    
    def run_full_analysis(self, target_area: Optional[Dict] = None) -> Dict:
        """
        Run complete intervention analysis
        
        Args:
            target_area: Optional target area characteristics
            
        Returns:
            Dictionary with complete analysis results
        """
        if target_area is None:
            # Create sample target area
            target_area = {
                'population': 8500,
                'unemployment_rate': 8.5,
                'crime_rate': 75,
                'social_trust_pre': 5.8,
                'community_cohesion_pre': 6.1,
                'volunteer_rate_pre': 15,
                'education_attainment': 68,
                'income_deprivation': 25,
                'ethnic_diversity': 35,
                'age_dependency_ratio': 0.65,
                'housing_stress': 40
            }
        
        # Find similar cases
        similar_cases = self.find_similar_cases(target_area, n_similar=20)
        
        # Get recommendations
        recommendations = self.recommend_interventions(target_area, n_recommendations=5)
        
        # Summary statistics
        summary = {
            'total_cases_in_database': len(self.intervention_database),
            'intervention_types_available': self.intervention_database['intervention_type'].nunique(),
            'average_success_score': self.intervention_database['success_score'].mean(),
            'most_common_intervention': self.intervention_database['intervention_type'].mode().iloc[0],
            'top_performing_intervention': self.intervention_database.groupby('intervention_type')['success_score'].mean().idxmax()
        }
        
        return {
            'target_area': target_area,
            'similar_cases': similar_cases,
            'recommendations': recommendations,
            'summary': summary,
            'analysis_timestamp': pd.Timestamp.now()
        }
