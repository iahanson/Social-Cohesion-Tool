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
from .unified_data_connector import UnifiedDataConnector
warnings.filterwarnings('ignore')

class SentimentMapping:
    """Sentiment and social trust mapping with GIS integration"""
    
    def __init__(self, data_connector=None):
        self.data_config = get_data_config()
        self.map_center = [51.5074, -0.1278]  # London coordinates
        self.map_zoom = 10
        self.data_connector = data_connector  # Use shared data connector if provided
        
    def load_data(self, n_areas: int = 50) -> pd.DataFrame:
        """
        Load data from configured sources (real or dummy)
        
        Args:
            n_areas: Number of areas to generate (for dummy data)
            
        Returns:
            DataFrame with sentiment and trust data
        """
        # Always try to use real data first (Good Neighbours, IMD, Population)
        if (use_real_data('good_neighbours') or use_real_data('imd_data') or 
            use_real_data('community_life_survey') or use_real_data('ons_census')):
            return self._load_real_data()
        else:
            print("Using sample data for sentiment mapping (configured for dummy data)")
            return self.generate_sample_data(n_areas)
    
    def _load_real_data(self) -> pd.DataFrame:
        """
        Load real data from configured sources using UnifiedDataConnector
        """
        try:
            print("🔄 Loading real data for sentiment mapping...")
            
            # Use shared data connector or create new one if not provided
            if self.data_connector is None:
                self.data_connector = UnifiedDataConnector(auto_load=True)
            else:
                # Ensure data is loaded in shared connector
                if self.data_connector.good_neighbours_data is None:
                    self.data_connector._load_data_sources()
            
            # Get the real data sources
            good_neighbours_data = self.data_connector.good_neighbours_data
            imd_data = self.data_connector.imd_data
            population_data = self.data_connector.msoa_population_data
            
            if good_neighbours_data is None:
                print("⚠️ No Good Neighbours data available, using sample data")
                return self.generate_sample_data(50)
            
            print(f"✅ Loaded real data: {len(good_neighbours_data)} MSOAs")
            
            # Start with Good Neighbours data as the base
            combined_data = good_neighbours_data.copy()
            
            # Add IMD data if available
            if imd_data is not None:
                print(f"✅ Adding IMD data: {len(imd_data)} MSOAs")
                # Merge IMD data
                combined_data = combined_data.merge(
                    imd_data[['msoa_code', 'msoa_imd_decile', 'msoa_imd_rank']], 
                    on='msoa_code', 
                    how='left'
                )
            else:
                print("⚠️ No IMD data available")
                # Add dummy IMD data
                combined_data['msoa_imd_decile'] = np.random.randint(1, 11, len(combined_data))
                combined_data['msoa_imd_rank'] = np.random.randint(1, 10000, len(combined_data))
            
            # Add population data if available
            if population_data is not None:
                print(f"✅ Adding population data: {len(population_data)} MSOAs")
                # Merge population data
                combined_data = combined_data.merge(
                    population_data[['msoa_code', 'total_population']], 
                    on='msoa_code', 
                    how='left'
                )
                # Rename to 'population' for compatibility with existing code
                combined_data['population'] = combined_data['total_population']
            else:
                print("⚠️ No population data available")
                # Add dummy population data
                combined_data['total_population'] = np.random.randint(1000, 15000, len(combined_data))
                combined_data['population'] = combined_data['total_population']
            
            # Add geographic coordinates (dummy for now - would need real MSOA boundary data)
            combined_data['latitude'] = np.random.normal(51.5074, 0.1, len(combined_data))
            combined_data['longitude'] = np.random.normal(-0.1278, 0.1, len(combined_data))
            
            # Add local authority information (dummy for now)
            combined_data['local_authority'] = 'London Borough'  # Would need real LA mapping
            
            # Ensure we have the required columns for sentiment mapping
            required_columns = ['msoa_code', 'msoa_name', 'net_trust', 'msoa_imd_decile', 
                              'total_population', 'latitude', 'longitude', 'local_authority']
            
            for col in required_columns:
                if col not in combined_data.columns:
                    print(f"⚠️ Missing column {col}, adding dummy data")
                    if col == 'net_trust':
                        combined_data[col] = np.random.normal(0, 0.2, len(combined_data))
                    elif col == 'msoa_name':
                        combined_data[col] = combined_data['msoa_code'].apply(lambda x: f"MSOA {x}")
                    else:
                        combined_data[col] = np.random.random(len(combined_data))
            
            print(f"✅ Real data loaded successfully: {len(combined_data)} MSOAs")
            return combined_data
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print("🔄 Falling back to sample data")
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
        
        # Check if we have real data columns or dummy data columns
        has_real_data = 'net_trust' in df.columns
        
        if has_real_data:
            # Use real data columns for composite scores
            print("🔄 Calculating composite scores using real data columns...")
            
            # Social Trust Composite Score - use net_trust as primary indicator
            # Scale net_trust (-1 to 1) to 1-10 scale for consistency
            df['social_trust_composite'] = ((df['net_trust'] + 1) / 2) * 9 + 1
            
            # Community Cohesion Composite Score - derive from available data
            # Use trust and caution indicators to estimate cohesion
            if 'always_usually_trust' in df.columns and 'usually_almost_always_careful' in df.columns:
                # Higher trust + lower caution = higher cohesion
                trust_score = df['always_usually_trust'] / 100  # Convert percentage to 0-1
                caution_score = df['usually_almost_always_careful'] / 100  # Convert percentage to 0-1
                df['community_cohesion_composite'] = (trust_score + (1 - caution_score)) * 5  # Scale to 1-10
            else:
                # Fallback: use net_trust as cohesion proxy
                df['community_cohesion_composite'] = df['social_trust_composite']
            
            # Sentiment Composite Score - derive from trust and deprivation
            if 'msoa_imd_decile' in df.columns:
                # Higher trust + lower deprivation = higher sentiment
                # IMD decile: 1 = most deprived, 10 = least deprived
                deprivation_score = (11 - df['msoa_imd_decile']) / 10  # Invert and normalize
                trust_score = df['social_trust_composite'] / 10  # Normalize trust to 0-1
                df['sentiment_composite'] = (trust_score + (1 - deprivation_score)) * 5  # Scale to 1-10
            else:
                # Fallback: use trust as sentiment proxy
                df['sentiment_composite'] = df['social_trust_composite']
            
            # Deprivation Composite Score (inverted - higher values = less deprived)
            if 'msoa_imd_decile' in df.columns:
                # IMD decile: 1 = most deprived, 10 = least deprived
                # Convert to 1-10 scale where 10 = least deprived
                df['deprivation_composite'] = df['msoa_imd_decile']
            else:
                # Fallback: generate dummy deprivation data
                df['deprivation_composite'] = np.random.randint(1, 11, len(df))
            
            # Overall Social Cohesion Score
            df['overall_cohesion_score'] = (
                df['social_trust_composite'] * 0.3 +
                df['community_cohesion_composite'] * 0.3 +
                df['sentiment_composite'] * 0.2 +
                df['deprivation_composite'] / 10 * 0.2
            )
            
            print(f"✅ Composite scores calculated using real data")
            
        else:
            # Use dummy data columns for composite scores (original logic)
            print("🔄 Calculating composite scores using dummy data columns...")
            
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
            
            print(f"✅ Composite scores calculated using dummy data")
        
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
        # Select indicators for correlation analysis - use available columns
        base_indicators = [
            'social_trust_composite', 'community_cohesion_composite', 
            'sentiment_composite', 'deprivation_composite'
        ]
        
        # Add additional indicators if they exist in the data
        additional_indicators = [
            'crime_rate', 'unemployment_rate', 'housing_stress',
            'volunteer_participation', 'community_events_attendance',
            'net_trust', 'msoa_imd_decile', 'population'
        ]
        
        # Build list of available indicators
        indicators = base_indicators.copy()
        for indicator in additional_indicators:
            if indicator in data.columns:
                indicators.append(indicator)
        
        # Ensure we have at least the base indicators
        available_indicators = [ind for ind in indicators if ind in data.columns]
        
        if len(available_indicators) < 2:
            print("⚠️ Not enough indicators for correlation analysis")
            # Create a simple correlation with just the base indicators
            available_indicators = [ind for ind in base_indicators if ind in data.columns]
        
        # Calculate correlation matrix
        corr_matrix = data[available_indicators].corr()
        
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
    
    def _safe_population_size(self, data: pd.DataFrame) -> list:
        """
        Safely calculate population-based marker sizes, handling NaN values
        
        Args:
            data: DataFrame with population data
            
        Returns:
            List of safe size values for Plotly markers
        """
        if 'population' in data.columns:
            # Handle NaN values by filling with median population
            population = data['population'].fillna(data['population'].median())
            # Convert to size (divide by 1000) and ensure minimum size
            sizes = (population / 1000).clip(lower=5)  # Minimum size of 5
            return sizes.tolist()
        else:
            # Default size if no population data
            return [10] * len(data)
    
    def create_deprivation_trust_scatter(self, data: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot showing relationship between deprivation and trust
        
        Args:
            data: DataFrame with deprivation and trust data
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Color by local authority (if available) or use a single color
        if 'local_authority' in data.columns:
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
                        size=self._safe_population_size(authority_data),
                        sizemode='diameter',
                        sizemin=5,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    text=authority_data['msoa_name'] if 'msoa_name' in authority_data.columns else authority_data['msoa_code'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Deprivation: %{x:.1f}<br>' +
                                'Trust: %{y:.1f}<br>' +
                                '<extra></extra>'
                ))
        else:
            # Fallback: use all data as one group
            fig.add_trace(go.Scatter(
                x=data['deprivation_composite'],
                y=data['social_trust_composite'],
                mode='markers',
                name='All Areas',
                marker=dict(
                    size=self._safe_population_size(data),
                    sizemode='diameter',
                    sizemin=5,
                    color='blue',
                    opacity=0.7
                ),
                text=data['msoa_name'] if 'msoa_name' in data.columns else data['msoa_code'],
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
