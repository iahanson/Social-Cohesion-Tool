"""
Early Warning System for Social Tension
Uses anomaly detection and clustering to identify areas with rising tensions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
from .data_config import get_data_config, use_real_data, use_dummy_data
warnings.filterwarnings('ignore')

class EarlyWarningSystem:
    """Early warning system for detecting social tension hotspots"""
    
    def __init__(self):
        self.data_config = get_data_config()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=0.95)
        self.risk_threshold = 0.7
        
    def load_data(self, n_areas: int = 100) -> pd.DataFrame:
        """
        Load data from configured sources (real or dummy)
        
        Args:
            n_areas: Number of areas to generate (for dummy data)
            
        Returns:
            DataFrame with social tension indicators
        """
        # Check if we should use real data
        if use_real_data('community_life_survey') or use_real_data('crime_data'):
            return self._load_real_data()
        else:
            print("Using sample data for early warning system (configured for dummy data)")
            return self.generate_sample_data(n_areas)
    
    def _load_real_data(self) -> pd.DataFrame:
        """
        Load real data from configured sources
        This would integrate with actual data connectors
        """
        # TODO: Implement real data loading when connectors are available
        print("Real data loading not yet implemented - using sample data")
        return self.generate_sample_data(100)
    
    def generate_sample_data(self, n_areas: int = 100) -> pd.DataFrame:
        """
        Generate sample data for demonstration purposes
        In production, this would connect to real data sources
        """
        np.random.seed(42)
        
        # Generate synthetic MSOA data
        data = {
            'msoa_code': [f'E0200{i:04d}' for i in range(1, n_areas + 1)],
            'local_authority': np.random.choice(['Camden', 'Westminster', 'Kensington', 'Hammersmith'], n_areas),
            'population': np.random.normal(8000, 2000, n_areas).astype(int),
            'unemployment_rate': np.random.beta(2, 8, n_areas) * 20,  # 0-20%
            'crime_rate': np.random.gamma(2, 50, n_areas),  # crimes per 1000
            'social_trust_score': np.random.normal(6.5, 1.5, n_areas),  # 1-10 scale
            'community_cohesion': np.random.normal(7.0, 1.2, n_areas),  # 1-10 scale
            'economic_uncertainty': np.random.beta(3, 7, n_areas) * 10,  # 0-10 scale
            'sentiment_score': np.random.normal(0, 1, n_areas),  # -3 to +3
            'social_media_mentions': np.random.poisson(50, n_areas),
            'negative_sentiment_ratio': np.random.beta(2, 8, n_areas),  # 0-1
            'community_events': np.random.poisson(5, n_areas),
            'volunteer_rate': np.random.beta(3, 7, n_areas) * 30,  # 0-30%
            'housing_stress': np.random.beta(4, 6, n_areas) * 100,  # 0-100%
            'education_attainment': np.random.normal(65, 15, n_areas),  # % with qualifications
        }
        
        df = pd.DataFrame(data)
        
        # Add some correlation between variables
        df['social_trust_score'] = df['social_trust_score'] - df['unemployment_rate'] * 0.1
        df['community_cohesion'] = df['community_cohesion'] - df['crime_rate'] * 0.02
        df['sentiment_score'] = df['sentiment_score'] - df['negative_sentiment_ratio'] * 2
        
        return df
    
    def calculate_risk_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for each area based on multiple indicators
        
        Args:
            data: DataFrame with area-level indicators
            
        Returns:
            DataFrame with added risk scores and components
        """
        df = data.copy()
        
        # Define risk indicators (higher values = higher risk)
        risk_indicators = [
            'unemployment_rate',
            'crime_rate', 
            'economic_uncertainty',
            'negative_sentiment_ratio',
            'housing_stress'
        ]
        
        # Define protective factors (higher values = lower risk)
        protective_indicators = [
            'social_trust_score',
            'community_cohesion',
            'volunteer_rate',
            'education_attainment',
            'community_events'
        ]
        
        # Normalize indicators to 0-1 scale
        for indicator in risk_indicators:
            df[f'{indicator}_normalized'] = (df[indicator] - df[indicator].min()) / (df[indicator].max() - df[indicator].min())
        
        for indicator in protective_indicators:
            df[f'{indicator}_normalized'] = 1 - (df[indicator] - df[indicator].min()) / (df[indicator].max() - df[indicator].min())
        
        # Calculate weighted risk score
        risk_weights = {
            'unemployment_rate_normalized': 0.2,
            'crime_rate_normalized': 0.2,
            'economic_uncertainty_normalized': 0.15,
            'negative_sentiment_ratio_normalized': 0.15,
            'housing_stress_normalized': 0.1,
            'social_trust_score_normalized': 0.1,
            'community_cohesion_normalized': 0.05,
            'volunteer_rate_normalized': 0.025,
            'education_attainment_normalized': 0.025
        }
        
        df['risk_score'] = 0
        for indicator, weight in risk_weights.items():
            df['risk_score'] += df[indicator] * weight
        
        # Add risk level categories
        df['risk_level'] = pd.cut(df['risk_score'], 
                                 bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                 labels=['Low', 'Medium', 'High', 'Critical'])
        
        return df
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalous areas using isolation forest
        
        Args:
            data: DataFrame with risk indicators
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        df = data.copy()
        
        # Select features for anomaly detection
        features = [
            'unemployment_rate', 'crime_rate', 'social_trust_score',
            'community_cohesion', 'economic_uncertainty', 'sentiment_score',
            'negative_sentiment_ratio', 'housing_stress'
        ]
        
        # Prepare data for anomaly detection
        X = df[features].fillna(df[features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit anomaly detector
        anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
        anomaly_scores_continuous = self.anomaly_detector.decision_function(X_scaled)
        
        df['anomaly_score'] = anomaly_scores_continuous
        df['is_anomaly'] = anomaly_scores == -1
        
        return df
    
    def cluster_areas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster areas based on social indicators
        
        Args:
            data: DataFrame with social indicators
            
        Returns:
            DataFrame with cluster assignments
        """
        df = data.copy()
        
        # Select features for clustering
        features = [
            'unemployment_rate', 'crime_rate', 'social_trust_score',
            'community_cohesion', 'economic_uncertainty', 'housing_stress',
            'volunteer_rate', 'education_attainment'
        ]
        
        # Prepare data for clustering
        X = df[features].fillna(df[features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Perform clustering
        clusters = self.clusterer.fit_predict(X_pca)
        
        df['cluster'] = clusters
        df['cluster_label'] = df['cluster'].map({
            -1: 'Outlier',
            0: 'Stable',
            1: 'At Risk',
            2: 'High Risk'
        }).fillna('Other')
        
        return df
    
    def generate_alerts(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate alerts for high-risk areas
        
        Args:
            data: DataFrame with risk scores and anomaly detection
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Critical risk areas
        critical_areas = data[data['risk_level'] == 'Critical']
        for _, area in critical_areas.iterrows():
            alerts.append({
                'type': 'CRITICAL_RISK',
                'msoa_code': area['msoa_code'],
                'local_authority': area['local_authority'],
                'risk_score': area['risk_score'],
                'message': f"Critical risk detected in {area['msoa_code']} ({area['local_authority']})",
                'priority': 'HIGH',
                'recommended_actions': [
                    'Immediate community engagement',
                    'Increased police presence',
                    'Social services review',
                    'Community event funding'
                ]
            })
        
        # Anomalous areas
        anomalous_areas = data[data['is_anomaly'] == True]
        for _, area in anomalous_areas.iterrows():
            alerts.append({
                'type': 'ANOMALY_DETECTED',
                'msoa_code': area['msoa_code'],
                'local_authority': area['local_authority'],
                'anomaly_score': area['anomaly_score'],
                'message': f"Unusual patterns detected in {area['msoa_code']} ({area['local_authority']})",
                'priority': 'MEDIUM',
                'recommended_actions': [
                    'Monitor closely',
                    'Community survey',
                    'Local stakeholder consultation'
                ]
            })
        
        return alerts
    
    def get_risk_factors(self, data: pd.DataFrame, msoa_code: str) -> Dict:
        """
        Get detailed risk factors for a specific MSOA
        
        Args:
            data: DataFrame with risk data
            msoa_code: MSOA code to analyze
            
        Returns:
            Dictionary with risk factor breakdown
        """
        area_data = data[data['msoa_code'] == msoa_code]
        
        if area_data.empty:
            return {'error': 'MSOA not found'}
        
        area = area_data.iloc[0]
        
        risk_factors = {
            'msoa_code': msoa_code,
            'local_authority': area['local_authority'],
            'overall_risk_score': area['risk_score'],
            'risk_level': area['risk_level'],
            'top_risk_factors': [],
            'protective_factors': [],
            'recommendations': []
        }
        
        # Identify top risk factors
        risk_indicators = [
            'unemployment_rate', 'crime_rate', 'economic_uncertainty',
            'negative_sentiment_ratio', 'housing_stress'
        ]
        
        risk_scores = []
        for indicator in risk_indicators:
            normalized_value = area[f'{indicator}_normalized']
            risk_scores.append((indicator, normalized_value))
        
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        risk_factors['top_risk_factors'] = risk_scores[:3]
        
        # Identify protective factors
        protective_indicators = [
            'social_trust_score', 'community_cohesion', 'volunteer_rate',
            'education_attainment', 'community_events'
        ]
        
        protective_scores = []
        for indicator in protective_indicators:
            normalized_value = area[f'{indicator}_normalized']
            protective_scores.append((indicator, normalized_value))
        
        protective_scores.sort(key=lambda x: x[1], reverse=True)
        risk_factors['protective_factors'] = protective_scores[:3]
        
        # Generate recommendations based on risk factors
        recommendations = []
        if area['unemployment_rate'] > data['unemployment_rate'].quantile(0.8):
            recommendations.append('Employment support programs')
        if area['crime_rate'] > data['crime_rate'].quantile(0.8):
            recommendations.append('Community safety initiatives')
        if area['social_trust_score'] < data['social_trust_score'].quantile(0.2):
            recommendations.append('Community building activities')
        
        risk_factors['recommendations'] = recommendations
        
        return risk_factors
    
    def run_full_analysis(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete early warning analysis
        
        Args:
            data: Optional DataFrame, if None generates sample data
            
        Returns:
            Dictionary with complete analysis results
        """
        if data is None:
            data = self.load_data()
        
        # Calculate risk scores
        data_with_risk = self.calculate_risk_score(data)
        
        # Detect anomalies
        data_with_anomalies = self.detect_anomalies(data_with_risk)
        
        # Cluster areas
        data_with_clusters = self.cluster_areas(data_with_anomalies)
        
        # Generate alerts
        alerts = self.generate_alerts(data_with_clusters)
        
        # Summary statistics
        summary = {
            'total_areas': len(data_with_clusters),
            'critical_risk_areas': len(data_with_clusters[data_with_clusters['risk_level'] == 'Critical']),
            'high_risk_areas': len(data_with_clusters[data_with_clusters['risk_level'] == 'High']),
            'anomalous_areas': len(data_with_clusters[data_with_clusters['is_anomaly'] == True]),
            'total_alerts': len(alerts),
            'average_risk_score': data_with_clusters['risk_score'].mean(),
            'risk_distribution': data_with_clusters['risk_level'].value_counts().to_dict()
        }
        
        return {
            'data': data_with_clusters,
            'alerts': alerts,
            'summary': summary,
            'analysis_timestamp': pd.Timestamp.now()
        }
