"""
Sentiment & Social Trust Mapping
Spatial mapping of trust, sentiment, and socio-economic factors
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Optional, Tuple
import warnings
from .data_config import get_data_config, use_real_data, use_dummy_data
warnings.filterwarnings('ignore')

class SentimentMapping:
    """Sentiment and social trust mapping with GIS integration"""
    
    def __init__(self):
        self.data_config = get_data_config()
        self.map_center = [51.5074, -0.1278]  # London coordinates
        self.map_zoom = 10
        
    def load_data(self, n_areas: int = 50) -> pd.DataFrame:
        """
        Load data from configured sources (real or dummy)
        
        Args:
            n_areas: Number of areas to generate (for dummy data)
            
        Returns:
            DataFrame with sentiment and trust data
        """
        # Check if we should use real data
        if use_real_data('community_life_survey') or use_real_data('ons_census'):
            return self._load_real_data()
        else:
            print("Using sample data for sentiment mapping (configured for dummy data)")
            return self.generate_sample_data(n_areas)
    
    def _load_real_data(self) -> pd.DataFrame:
        """
        Load real data from configured sources
        This would integrate with actual data connectors
        """
        # TODO: Implement real data loading when connectors are available
        print("Real data loading not yet implemented - using sample data")
        return self.generate_sample_data(50)
    
    def generate_sample_data(self, n_areas: int = 50) -> pd.DataFrame:
        """
        Generate sample data for demonstration
        In production, this would connect to Community Life Survey and other sources
        """
        np.random.seed(42)
        
        # Generate synthetic MSOA data with geographical coordinates
        data = {
            'msoa_code': [f'E0200{i:04d}' for i in range(1, n_areas + 1)],
            'msoa_name': [f'MSOA {i:04d}' for i in range(1, n_areas + 1)],
            'local_authority': np.random.choice(['Camden', 'Westminster', 'Kensington', 'Hammersmith', 'Tower Hamlets'], n_areas),
            'latitude': np.random.normal(51.5074, 0.1, n_areas),
            'longitude': np.random.normal(-0.1278, 0.1, n_areas),
            'population': np.random.normal(8000, 2000, n_areas).astype(int),
            
            # Social trust indicators (1-10 scale)
            'trust_neighbors': np.random.normal(7.2, 1.5, n_areas),
            'trust_local_council': np.random.normal(5.8, 1.8, n_areas),
            'trust_police': np.random.normal(6.5, 1.6, n_areas),
            'trust_healthcare': np.random.normal(7.8, 1.2, n_areas),
            'trust_education': np.random.normal(7.0, 1.4, n_areas),
            
            # Community cohesion indicators
            'community_belonging': np.random.normal(6.8, 1.6, n_areas),
            'volunteer_participation': np.random.beta(3, 7, n_areas) * 30,  # 0-30%
            'community_events_attendance': np.random.poisson(3, n_areas),
            'local_friendships': np.random.normal(6.5, 1.8, n_areas),
            
            # Sentiment indicators
            'overall_satisfaction': np.random.normal(6.9, 1.4, n_areas),
            'economic_optimism': np.random.normal(5.5, 1.8, n_areas),
            'future_outlook': np.random.normal(6.2, 1.6, n_areas),
            'social_cohesion_score': np.random.normal(6.8, 1.3, n_areas),
            
            # Deprivation indicators
            'imd_rank': np.random.randint(1, 32844, n_areas),  # England has 32,844 LSOAs
            'income_deprivation': np.random.beta(2, 8, n_areas) * 100,
            'employment_deprivation': np.random.beta(2, 8, n_areas) * 100,
            'education_deprivation': np.random.beta(2, 8, n_areas) * 100,
            'health_deprivation': np.random.beta(2, 8, n_areas) * 100,
            
            # Crime and safety
            'crime_rate': np.random.gamma(2, 50, n_areas),
            'perceived_safety': np.random.normal(6.5, 1.7, n_areas),
            'anti_social_behavior': np.random.poisson(15, n_areas),
            
            # Economic indicators
            'unemployment_rate': np.random.beta(2, 8, n_areas) * 20,
            'household_income': np.random.normal(45000, 15000, n_areas),
            'housing_stress': np.random.beta(4, 6, n_areas) * 100,
        }
        
        df = pd.DataFrame(data)
        
        # Add correlations between variables
        df['trust_neighbors'] = df['trust_neighbors'] - df['crime_rate'] * 0.01
        df['community_belonging'] = df['community_belonging'] - df['income_deprivation'] * 0.02
        df['overall_satisfaction'] = df['overall_satisfaction'] - df['unemployment_rate'] * 0.05
        df['social_cohesion_score'] = (df['trust_neighbors'] + df['community_belonging'] + 
                                     df['volunteer_participation']/3) / 3
        
        # Ensure values are within reasonable bounds
        trust_cols = ['trust_neighbors', 'trust_local_council', 'trust_police', 
                     'trust_healthcare', 'trust_education', 'community_belonging',
                     'local_friendships', 'overall_satisfaction', 'economic_optimism',
                     'future_outlook', 'social_cohesion_score', 'perceived_safety']
        
        for col in trust_cols:
            df[col] = np.clip(df[col], 1, 10)
        
        return df
    
    def calculate_composite_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite scores for trust, cohesion, and sentiment
        
        Args:
            data: DataFrame with individual indicators
            
        Returns:
            DataFrame with composite scores
        """
        df = data.copy()
        
        # Social Trust Composite Score
        trust_indicators = [
            'trust_neighbors', 'trust_local_council', 'trust_police',
            'trust_healthcare', 'trust_education'
        ]
        df['social_trust_composite'] = df[trust_indicators].mean(axis=1)
        
        # Community Cohesion Composite Score
        cohesion_indicators = [
            'community_belonging', 'volunteer_participation', 
            'community_events_attendance', 'local_friendships'
        ]
        # Normalize volunteer participation to 1-10 scale
        df['volunteer_normalized'] = (df['volunteer_participation'] / 30) * 10
        df['events_normalized'] = np.clip(df['community_events_attendance'] / 5 * 10, 1, 10)
        
        df['community_cohesion_composite'] = df[['community_belonging', 'volunteer_normalized', 
                                               'events_normalized', 'local_friendships']].mean(axis=1)
        
        # Sentiment Composite Score
        sentiment_indicators = [
            'overall_satisfaction', 'economic_optimism', 'future_outlook'
        ]
        df['sentiment_composite'] = df[sentiment_indicators].mean(axis=1)
        
        # Deprivation Composite Score (inverted - higher values = less deprived)
        deprivation_indicators = [
            'income_deprivation', 'employment_deprivation', 
            'education_deprivation', 'health_deprivation'
        ]
        df['deprivation_composite'] = 100 - df[deprivation_indicators].mean(axis=1)
        
        # Overall Social Cohesion Score
        df['overall_cohesion_score'] = (
            df['social_trust_composite'] * 0.3 +
            df['community_cohesion_composite'] * 0.3 +
            df['sentiment_composite'] * 0.2 +
            df['deprivation_composite'] / 10 * 0.2
        )
        
        return df
    
    def create_trust_map(self, data: pd.DataFrame) -> folium.Map:
        """
        Create interactive map showing social trust levels
        
        Args:
            data: DataFrame with trust data and coordinates
            
        Returns:
            Folium map object
        """
        try:
            # Create base map with greyscale tiles for better contrast
            m = folium.Map(
                location=self.map_center,
                zoom_start=self.map_zoom,
                tiles='CartoDB positron'  # Light greyscale basemap
            )
            
            # Add trust data as circles
            for _, row in data.iterrows():
                try:
                    # Validate coordinates
                    lat, lon = row['latitude'], row['longitude']
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Color based on trust level
                    trust_score = row['social_trust_composite']
                    if trust_score >= 7:
                        color = 'green'
                    elif trust_score >= 5:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    # Size based on population
                    radius = max(5, min(20, row['population'] / 1000))
                    
                    # Create popup text safely
                    popup_text = f"""
                    <b>{row.get('msoa_name', 'Unknown')}</b><br>
                    MSOA: {row.get('msoa_code', 'N/A')}<br>
                    Local Authority: {row.get('local_authority', 'N/A')}<br>
                    Social Trust: {trust_score:.1f}/10<br>
                    Population: {row.get('population', 0):,}
                    """
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=folium.Popup(popup_text, max_width=200),
                        color='black',
                        weight=1,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
                    
                except Exception as e:
                    print(f"Error adding marker for row {row.get('msoa_code', 'unknown')}: {e}")
                    continue
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 90px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Social Trust Levels</b></p>
            <p><i class="fa fa-circle" style="color:green"></i> High (7+)</p>
            <p><i class="fa fa-circle" style="color:orange"></i> Medium (5-7)</p>
            <p><i class="fa fa-circle" style="color:red"></i> Low (<5)</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m
            
        except Exception as e:
            print(f"Error creating trust map: {e}")
            # Return a simple map as fallback
            return folium.Map(
                location=self.map_center,
                zoom_start=self.map_zoom,
                tiles='CartoDB positron'
            )
    
    def _create_alternative_map(self, data: pd.DataFrame) -> go.Figure:
        """
        Create alternative map visualization using Plotly
        
        Args:
            data: DataFrame with location and trust data
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.scatter_mapbox(
                data,
                lat='latitude',
                lon='longitude',
                color='social_trust_composite',
                size='population',
                hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                color_continuous_scale='RdYlGn',
                mapbox_style='carto-positron',  # Light greyscale basemap
                title='Social Trust Map (Alternative View)',
                zoom=10,
                center=dict(lat=self.map_center[0], lon=self.map_center[1])
            )
            
            fig.update_layout(
                height=500,
                margin=dict(r=0, t=30, l=0, b=0)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating alternative map: {e}")
            return None
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap of social indicators
        
        Args:
            data: DataFrame with social indicators
            
        Returns:
            Plotly figure object
        """
        # Select indicators for correlation analysis
        indicators = [
            'social_trust_composite', 'community_cohesion_composite', 
            'sentiment_composite', 'deprivation_composite',
            'crime_rate', 'unemployment_rate', 'housing_stress',
            'volunteer_participation', 'community_events_attendance'
        ]
        
        # Calculate correlation matrix
        corr_matrix = data[indicators].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix: Social Indicators',
            xaxis_title='Indicators',
            yaxis_title='Indicators',
            width=800,
            height=600
        )
        
        return fig
    
    def create_deprivation_trust_scatter(self, data: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot showing relationship between deprivation and trust
        
        Args:
            data: DataFrame with deprivation and trust data
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Color by local authority
        authorities = data['local_authority'].unique()
        colors = px.colors.qualitative.Set3
        
        for i, authority in enumerate(authorities):
            authority_data = data[data['local_authority'] == authority]
            
            fig.add_trace(go.Scatter(
                x=authority_data['deprivation_composite'],
                y=authority_data['social_trust_composite'],
                mode='markers',
                name=authority,
                marker=dict(
                    size=authority_data['population'] / 1000,
                    sizemode='diameter',
                    sizemin=5,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=authority_data['msoa_name'],
                hovertemplate='<b>%{text}</b><br>' +
                            'Deprivation: %{x:.1f}<br>' +
                            'Trust: %{y:.1f}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Social Trust vs Deprivation by Local Authority',
            xaxis_title='Deprivation Score (Higher = Less Deprived)',
            yaxis_title='Social Trust Score (1-10)',
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_cohesion_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """
        Create dashboard showing multiple cohesion indicators
        
        Args:
            data: DataFrame with cohesion data
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Social Trust Distribution', 'Community Cohesion Distribution',
                          'Sentiment Distribution', 'Overall Cohesion Score'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Social Trust Distribution
        fig.add_trace(
            go.Histogram(x=data['social_trust_composite'], name='Social Trust', nbinsx=20),
            row=1, col=1
        )
        
        # Community Cohesion Distribution
        fig.add_trace(
            go.Histogram(x=data['community_cohesion_composite'], name='Community Cohesion', nbinsx=20),
            row=1, col=2
        )
        
        # Sentiment Distribution
        fig.add_trace(
            go.Histogram(x=data['sentiment_composite'], name='Sentiment', nbinsx=20),
            row=2, col=1
        )
        
        # Overall Cohesion Score Distribution
        fig.add_trace(
            go.Histogram(x=data['overall_cohesion_score'], name='Overall Cohesion', nbinsx=20),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Social Cohesion Dashboard',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def analyze_area_profile(self, data: pd.DataFrame, msoa_code: str) -> Dict:
        """
        Analyze detailed profile for a specific area
        
        Args:
            data: DataFrame with area data
            msoa_code: MSOA code to analyze
            
        Returns:
            Dictionary with area profile
        """
        area_data = data[data['msoa_code'] == msoa_code]
        
        if area_data.empty:
            return {'error': 'MSOA not found'}
        
        area = area_data.iloc[0]
        
        # Calculate percentile rankings
        trust_percentile = (data['social_trust_composite'] < area['social_trust_composite']).mean() * 100
        cohesion_percentile = (data['community_cohesion_composite'] < area['community_cohesion_composite']).mean() * 100
        sentiment_percentile = (data['sentiment_composite'] < area['sentiment_composite']).mean() * 100
        
        profile = {
            'msoa_code': msoa_code,
            'msoa_name': area['msoa_name'],
            'local_authority': area['local_authority'],
            'population': int(area['population']),
            
            'scores': {
                'social_trust': {
                    'score': round(area['social_trust_composite'], 1),
                    'percentile': round(trust_percentile, 1),
                    'neighbors': round(area['trust_neighbors'], 1),
                    'council': round(area['trust_local_council'], 1),
                    'police': round(area['trust_police'], 1),
                    'healthcare': round(area['trust_healthcare'], 1),
                    'education': round(area['trust_education'], 1)
                },
                'community_cohesion': {
                    'score': round(area['community_cohesion_composite'], 1),
                    'percentile': round(cohesion_percentile, 1),
                    'belonging': round(area['community_belonging'], 1),
                    'volunteering': round(area['volunteer_participation'], 1),
                    'events': int(area['community_events_attendance']),
                    'friendships': round(area['local_friendships'], 1)
                },
                'sentiment': {
                    'score': round(area['sentiment_composite'], 1),
                    'percentile': round(sentiment_percentile, 1),
                    'satisfaction': round(area['overall_satisfaction'], 1),
                    'economic_optimism': round(area['economic_optimism'], 1),
                    'future_outlook': round(area['future_outlook'], 1)
                },
                'overall_cohesion': round(area['overall_cohesion_score'], 1)
            },
            
            'challenges': [],
            'strengths': [],
            'recommendations': []
        }
        
        # Identify challenges and strengths
        if area['social_trust_composite'] < data['social_trust_composite'].quantile(0.25):
            profile['challenges'].append('Low social trust levels')
        if area['community_cohesion_composite'] < data['community_cohesion_composite'].quantile(0.25):
            profile['challenges'].append('Weak community cohesion')
        if area['crime_rate'] > data['crime_rate'].quantile(0.75):
            profile['challenges'].append('High crime rates')
        
        if area['social_trust_composite'] > data['social_trust_composite'].quantile(0.75):
            profile['strengths'].append('High social trust')
        if area['volunteer_participation'] > data['volunteer_participation'].quantile(0.75):
            profile['strengths'].append('Strong volunteer participation')
        if area['community_events_attendance'] > data['community_events_attendance'].quantile(0.75):
            profile['strengths'].append('Active community engagement')
        
        # Generate recommendations
        if 'Low social trust' in profile['challenges']:
            profile['recommendations'].append('Implement community building programs')
        if 'Weak community cohesion' in profile['challenges']:
            profile['recommendations'].append('Increase funding for local events and activities')
        if 'High crime rates' in profile['challenges']:
            profile['recommendations'].append('Enhance community safety initiatives')
        
        return profile
    
    def run_full_analysis(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete sentiment and trust mapping analysis
        
        Args:
            data: Optional DataFrame, if None generates sample data
            
        Returns:
            Dictionary with complete analysis results
        """
        if data is None:
            data = self.load_data()
        
        # Calculate composite scores
        data_with_scores = self.calculate_composite_scores(data)
        
        # Create visualizations
        trust_map = self.create_trust_map(data_with_scores)
        correlation_heatmap = self.create_correlation_heatmap(data_with_scores)
        deprivation_scatter = self.create_deprivation_trust_scatter(data_with_scores)
        cohesion_dashboard = self.create_cohesion_dashboard(data_with_scores)
        
        # Summary statistics
        summary = {
            'total_areas': len(data_with_scores),
            'average_trust_score': data_with_scores['social_trust_composite'].mean(),
            'average_cohesion_score': data_with_scores['community_cohesion_composite'].mean(),
            'average_sentiment_score': data_with_scores['sentiment_composite'].mean(),
            'trust_range': {
                'min': data_with_scores['social_trust_composite'].min(),
                'max': data_with_scores['social_trust_composite'].max()
            },
            'cohesion_range': {
                'min': data_with_scores['community_cohesion_composite'].min(),
                'max': data_with_scores['community_cohesion_composite'].max()
            }
        }
        
        # Create alternative map visualization for better compatibility
        try:
            alternative_map = self._create_alternative_map(data_with_scores)
        except Exception as e:
            print(f"Error creating alternative map: {e}")
            alternative_map = None
        
        return {
            'data': data_with_scores,
            'visualizations': {
                'trust_map': trust_map,
                'alternative_map': alternative_map,
                'correlation_heatmap': correlation_heatmap,
                'deprivation_scatter': deprivation_scatter,
                'cohesion_dashboard': cohesion_dashboard
            },
            'summary': summary,
            'analysis_timestamp': pd.Timestamp.now()
        }
