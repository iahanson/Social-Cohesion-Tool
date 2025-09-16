"""
Streamlit Frontend for Social Cohesion Monitoring System
Main dashboard integrating all components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
load_dotenv()

# Try to import streamlit_folium, fallback if not available
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("streamlit-folium not installed. Maps will be displayed as static images. Install with: pip install streamlit-folium")

# Try to import geopandas for LAD boundaries
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("geopandas not available. LAD boundaries will be disabled.")

# Import our custom modules
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator
from src.alert_system import AlertSystem
from src.unified_data_connector import UnifiedDataConnector
from src.genai_text_analyzer import GenAITextAnalyzer
from src.locality_mapper import LocalityMapper

# Page configuration
st.set_page_config(
    page_title="Social Cohesion Monitoring System",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with shared data connector
if 'unified_data_connector' not in st.session_state:
    st.session_state.unified_data_connector = UnifiedDataConnector()

if 'early_warning_system' not in st.session_state:
    st.session_state.early_warning_system = EarlyWarningSystem(st.session_state.unified_data_connector)
if 'sentiment_mapping' not in st.session_state:
    st.session_state.sentiment_mapping = SentimentMapping(st.session_state.unified_data_connector)
if 'intervention_tool' not in st.session_state:
    st.session_state.intervention_tool = InterventionTool()
if 'engagement_simulator' not in st.session_state:
    st.session_state.engagement_simulator = EngagementSimulator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()

def find_lad_by_coordinates(clicked_coords: List[float], lads_gdf) -> Optional[str]:
    """Find which LAD contains the clicked coordinates"""
    try:
        from shapely.geometry import Point
        
        if not clicked_coords or len(clicked_coords) < 2:
            return None
        
        click_point = Point(clicked_coords[1], clicked_coords[0])  # lat, lon
        
        # Check which LAD geometry contains this point
        for idx, row in lads_gdf.iterrows():
            if row.geometry.contains(click_point):
                # Find the LAD name column
                for col in ['LAD24NM', 'LAD23NM', 'LAD22NM', 'LAD_NAME', 'Name', 'LADNM', 'LAD_NM', 'NAME']:
                    if col in row:
                        return row[col]
        return None
    except Exception as e:
        st.write(f"🔍 Debug - Error finding LAD by coordinates: {e}")
        return None

def get_lad_comprehensive_data(lad_name: str, connector) -> Dict[str, Any]:
    """Get comprehensive data for a specific LAD from all available sources"""
    data = {
        'lad_name': lad_name,
        'imd_data': None,
        'good_neighbours_data': None,
        'population_data': None,
        'community_survey_data': None,
        'unemployment_data': None,
        'msoa_count': 0,
        'lsoa_count': 0
    }
    
    try:
        # Get Community Life Survey data for this LAD
        if connector.community_survey_data is not None:
            lad_column = connector.community_survey_data.columns[1]  # Column B should be LAD names
            lad_survey_data = connector.community_survey_data[connector.community_survey_data[lad_column] == lad_name]
            if not lad_survey_data.empty:
                data['community_survey_data'] = lad_survey_data
        
        # Get MSOA-level data for this LAD
        if connector.msoa_population_data is not None:
            # Filter MSOAs that belong to this LAD
            lad_msoas = connector.msoa_population_data[connector.msoa_population_data['msoa_name'] == lad_name]
            data['msoa_count'] = len(lad_msoas)
            
            if not lad_msoas.empty:
                data['population_data'] = lad_msoas
                
                # Get IMD data for these MSOAs
                if connector.imd_data is not None:
                    msoa_codes = lad_msoas['msoa_code'].tolist()
                    lad_imd_data = connector.imd_data[connector.imd_data['msoa_code'].isin(msoa_codes)]
                    if not lad_imd_data.empty:
                        data['imd_data'] = lad_imd_data
                
                # Get Good Neighbours data for these MSOAs
                if connector.good_neighbours_data is not None:
                    msoa_codes = lad_msoas['msoa_code'].tolist()
                    lad_gn_data = connector.good_neighbours_data[connector.good_neighbours_data['msoa_code'].isin(msoa_codes)]
                    if not lad_gn_data.empty:
                        data['good_neighbours_data'] = lad_gn_data
        
        # Get unemployment data for this LAD
        unemployment_data = connector.get_unemployment_by_lad(lad_name)
        if unemployment_data:
            data['unemployment_data'] = unemployment_data
        
        # Estimate LSOA count (roughly 4-8 LSOAs per MSOA)
        data['lsoa_count'] = data['msoa_count'] * 6  # Average estimate
        
    except Exception as e:
        st.error(f"Error getting data for {lad_name}: {e}")
    
    return data

def create_interactive_uk_map():
    """Create an interactive UK map with LAD boundaries and risk choropleth using Folium and GeoPandas"""
    if not FOLIUM_AVAILABLE or not GEOPANDAS_AVAILABLE:
        st.error("Folium or GeoPandas not available. Cannot create interactive map.")
        return None, None
    
    try:
        # Try to use GeoJSON file first for real boundaries
        geojson_path = 'data/Local_Authority_Districts.geojson'
        csv_path = 'data/Local_Authority_Districts_May_2023.csv'
        
        lads = None
        
        # Method 1: Try to load from GeoJSON file
        try:
            lads = gpd.read_file(geojson_path)
            print(f"✅ Loaded {len(lads)} LADs from GeoJSON with real boundaries")
            
            # Check if geometries are valid
            valid_count = lads.geometry.is_valid.sum()
            if valid_count == 0:
                print("⚠️ GeoJSON has no valid geometries, falling back to CSV")
                lads = None
            else:
                print(f"✅ Found {valid_count} valid geometries in GeoJSON")
                
        except Exception as e:
            print(f"⚠️ Could not load GeoJSON: {e}")
            lads = None
        
        # Method 2: Fallback to CSV with improved boundaries
        if lads is None:
            try:
                print("🔄 Loading LAD data from CSV and creating improved boundaries...")
                df = pd.read_csv(csv_path)
            
                # Create more realistic boundaries using multiple circles/ellipses
                from shapely.geometry import Point, Polygon
                import numpy as np
                
                geometries = []
                for idx, row in df.iterrows():
                    center = Point(row['LONG'], row['LAT'])
                    
                    # Calculate radius based on area (if available) or use default
                    if 'Shape__Area' in df.columns:
                        # Convert area to approximate radius (assuming roughly circular)
                        area = row['Shape__Area']
                        radius = np.sqrt(area / np.pi) * 0.00001  # Scale factor for map coordinates
                        radius = max(0.01, min(0.05, radius))  # Clamp between reasonable bounds
                    else:
                        radius = 0.02  # Default radius
                    
                    # Create simple oval shape
                    # Use different radii for x and y to create an oval
                    radius_x = radius * 1.2  # Slightly wider
                    radius_y = radius * 0.8  # Slightly narrower
                    
                    # Create oval using ellipse approximation
                    # Generate points around an ellipse
                    num_points = 16  # More points for smoother oval
                    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                    
                    oval_points = []
                    for angle in angles:
                        x = center.x + radius_x * np.cos(angle)
                        y = center.y + radius_y * np.sin(angle)
                        oval_points.append((x, y))
                    
                    # Create oval polygon
                    try:
                        oval = Polygon(oval_points)
                        if oval.is_valid:
                            geometries.append(oval)
                        else:
                            # Fallback to simple circle if oval is invalid
                            geometries.append(center.buffer(radius))
                    except:
                        # Fallback to simple circle if oval creation fails
                        geometries.append(center.buffer(radius))
                
                # Create GeoDataFrame with improved boundaries
                lads = gpd.GeoDataFrame(df, geometry=geometries, crs='EPSG:4326')
                print(f"✅ Created {len(lads)} LADs with improved boundaries from CSV")
                
            except Exception as e:
                st.error(f"Failed to load CSV data: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                return None, None
        
        # Initialize a folium map centered on the UK
        m = folium.Map(
            location=[52.5, -1.5], 
            zoom_start=6, 
            tiles='cartodbpositron'
        )
        
        
        # Add LAD boundaries
        if lads is not None:
            # Determine the LAD name column - check all possible column names
            lad_name_col = None
            possible_cols = ['LAD24NM', 'LAD23NM', 'LAD22NM', 'LAD_NAME', 'Name', 'LADNM', 'LAD_NM', 'NAME']
            
            for col in possible_cols:
                if col in lads.columns:
                    lad_name_col = col
                    break
            
            if lad_name_col:
                # Get real risk data from Early Warning System
                try:
                    # Use the shared Early Warning System instance from session state
                    if 'early_warning_system' not in st.session_state:
                        from src.early_warning_system import EarlyWarningSystem
                        st.session_state.early_warning_system = EarlyWarningSystem(st.session_state.unified_data_connector)
                    
                    ew_data = st.session_state.early_warning_system.run_full_analysis()
                    
                    if ew_data and 'data' in ew_data:
                        risk_df = ew_data['data']
                        
                        # Initialize with default values
                        lads['risk_score'] = 0.5  # Default risk score
                        lads['risk_level'] = 'Medium'  # Default risk level
                        
                        # Map risk data to LADs
                        matched_count = 0
                        for idx, row in lads.iterrows():
                            lad_name = row.get('LAD24NM', '')
                            
                            # Look for exact match first
                            exact_match = risk_df[risk_df['area_name'].str.lower() == lad_name.lower()]
                            if not exact_match.empty:
                                risk_row = exact_match.iloc[0]
                                lads.at[idx, 'risk_score'] = risk_row.get('risk_score', 0.5)
                                lads.at[idx, 'risk_level'] = risk_row.get('risk_level', 'Medium')
                                matched_count += 1
                            else:
                                # Try partial matching for common LAD name variations
                                lad_parts = lad_name.split()
                                for part in lad_parts:
                                    if len(part) > 3:  # Only try meaningful parts
                                        partial_match = risk_df[risk_df['area_name'].str.contains(part, case=False, na=False)]
                                        if not partial_match.empty:
                                            risk_row = partial_match.iloc[0]
                                            lads.at[idx, 'risk_score'] = risk_row.get('risk_score', 0.5)
                                            lads.at[idx, 'risk_level'] = risk_row.get('risk_level', 'Medium')
                                            matched_count += 1
                                            break
                        
                        print(f"✅ Mapped Early Warning System risk data to {matched_count}/{len(lads)} LADs")
                    else:
                        print("⚠️ No Early Warning System data available, using default values")
                        lads['risk_score'] = 0.5
                        lads['risk_level'] = 'Medium'
                        
                except Exception as e:
                    print(f"⚠️ Error loading Early Warning System data: {e}")
                    # Fallback to sample data for demonstration
                    np.random.seed(42)  # For consistent results
                    lads['risk_score'] = np.random.uniform(0, 1, len(lads))
                    lads['risk_level'] = lads['risk_score'].apply(
                        lambda x: 'Critical' if x >= 0.8 else 'High' if x >= 0.6 else 'Medium' if x >= 0.4 else 'Low'
                    )
                
                
                # Define color mapping for risk levels
                def get_risk_color(risk_level):
                    color_map = {
                        'Low': '#2E8B57',      # Sea Green
                        'Medium': '#FFD700',    # Gold
                        'High': '#FF8C00',      # Dark Orange
                        'Critical': '#DC143C'   # Crimson
                    }
                    return color_map.get(risk_level, '#808080')  # Default gray
                
                # Check if we have valid geometries
                valid_count = lads.geometry.is_valid.sum()
                
                if valid_count == 0:
                    st.error("❌ No valid geometries found!")
                    return m
                
                # Try a simpler approach first - just add basic boundaries
                try:
                    # Add basic LAD boundaries first
                    folium.GeoJson(
                        lads,
                        name='LAD Boundaries (Basic)',
                        style_function=lambda x: {
                            'color': 'red',
                            'weight': 3,
                            'fillOpacity': 0.5,
                            'fillColor': 'yellow'
                        }
                    ).add_to(m)
                    
                    # Now try the risk choropleth with improved click handling
                    geojson_layer = folium.GeoJson(
                        lads,
                        name='LAD Risk Choropleth',
                        style_function=lambda feature: {
                            'color': '#000000',
                            'weight': 2,
                            'fillOpacity': 0.7,
                            'fillColor': get_risk_color(feature['properties']['risk_level'])
                        },
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=[lad_name_col, 'risk_score', 'risk_level'], 
                            aliases=['LAD Name', 'Risk Score', 'Risk Level']
                        ),
                        popup=folium.features.GeoJsonPopup(
                            fields=[lad_name_col, 'risk_score', 'risk_level'],
                            aliases=['LAD Name', 'Risk Score', 'Risk Level'],
                            localize=True,
                            labels=True
                        )
                    )
                    
                    # Add click event handler
                    geojson_layer.add_child(
                        folium.ClickForMarker(
                            popup=f"Clicked LAD: {lad_name_col}"
                        )
                    )
                    
                    geojson_layer.add_to(m)
                    
                except Exception as e:
                    st.error(f"❌ Error adding LAD boundaries: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Try even simpler approach
                    try:
                        st.info("🔄 Trying fallback approach...")
                        folium.GeoJson(
                            lads,
                            name='LAD Fallback',
                            style_function=lambda x: {
                                'color': 'blue',
                                'weight': 1,
                                'fillOpacity': 0.3,
                                'fillColor': 'lightblue'
                            }
                        ).add_to(m)
                    except Exception as e2:
                        st.error(f"❌ Fallback also failed: {e2}")
                
                # Add legend with better sizing
                legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 220px; height: 140px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:12px; padding: 12px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2)">
                <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 14px;">Risk Level Legend</p>
                <p style="margin: 4px 0; line-height: 1.3;"><span style="display: inline-block; width: 12px; height: 12px; background-color: #2E8B57; margin-right: 8px;"></span> Low Risk (0.0-0.4)</p>
                <p style="margin: 4px 0; line-height: 1.3;"><span style="display: inline-block; width: 12px; height: 12px; background-color: #FFD700; margin-right: 8px;"></span> Medium Risk (0.4-0.6)</p>
                <p style="margin: 4px 0; line-height: 1.3;"><span style="display: inline-block; width: 12px; height: 12px; background-color: #FF8C00; margin-right: 8px;"></span> High Risk (0.6-0.8)</p>
                <p style="margin: 4px 0; line-height: 1.3;"><span style="display: inline-block; width: 12px; height: 12px; background-color: #DC143C; margin-right: 8px;"></span> Critical Risk (0.8-1.0)</p>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                
            else:
                st.error(f"Could not find LAD name column. Available columns: {list(lads.columns)}")
                # Add boundaries without tooltips as fallback
                folium.GeoJson(
                    lads,
                    name='LAD Boundaries',
                    style_function=lambda x: {
                        'color': '#000000',
                        'weight': 2,
                        'fillOpacity': 0.3,
                        'fillColor': 'lightblue'
                    }
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        
        return m, lads
        
    except Exception as e:
        st.error(f"Error creating interactive map: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏘️ Social Cohesion Monitoring System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize page in session state if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Dashboard Overview"
    
    page_options = [
        "📊 Dashboard Overview",
        "🚨 Early Warning System",
        "🗺️ Sentiment & Trust Mapping",
        "🤝 Good Neighbours Trust Data",
        "📋 Community Life Survey",
        "💡 Intervention Recommendations",
        "🎯 Engagement Simulator",
        "📧 Alert Management",
        "🤖 GenAI Text Analysis",
        "⚙️ System Settings"
    ]
    
    try:
        current_index = page_options.index(st.session_state.current_page)
    except ValueError:
        current_index = 0
    
    # Use a session state-based key that's stable
    if 'nav_key' not in st.session_state:
        st.session_state.nav_key = f"main_nav_{len(page_options)}"
    
    page = st.sidebar.selectbox(
        "Choose a component:",
        page_options,
        key=st.session_state.nav_key,
        index=current_index
    )
    
    # Update session state
    st.session_state.current_page = page
    
    # Route to appropriate page
    if page == "📊 Dashboard Overview":
        dashboard_overview()
    elif page == "🚨 Early Warning System":
        early_warning_page()
    elif page == "🗺️ Sentiment & Trust Mapping":
        sentiment_mapping_page()
    elif page == "🤝 Good Neighbours Trust Data":
        good_neighbours_page()
    elif page == "📋 Community Life Survey":
        community_survey_page()
    elif page == "💡 Intervention Recommendations":
        intervention_page()
    elif page == "🎯 Engagement Simulator":
        engagement_simulator_page()
    elif page == "📧 Alert Management":
        alert_management_page()
    elif page == "🤖 GenAI Text Analysis":
        genai_text_analysis_page()
    elif page == "⚙️ System Settings":
        settings_page()

def dashboard_overview():
    """Dashboard overview page"""
    st.header("📊 System Overview")
    
    # Generate sample data for overview
    with st.spinner("Loading system data..."):
        # Early warning data
        ew_data = st.session_state.early_warning_system.run_full_analysis()
        
        # Sentiment mapping data
        sm_data = st.session_state.sentiment_mapping.run_full_analysis()
        
        # Intervention data
        int_data = st.session_state.intervention_tool.run_full_analysis()
        
        # Good Neighbours social trust data
        gn_data = st.session_state.unified_data_connector.load_good_neighbours_data()
        gn_summary = st.session_state.unified_data_connector.get_good_neighbours_summary()
        
        # Community Life Survey data for LAD-level insights
        survey_data = st.session_state.unified_data_connector.get_community_survey_data()
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="Areas Monitored",
            value=ew_data['summary']['total_areas'],
            delta=f"{ew_data['summary']['critical_risk_areas']} critical"
        )
    
    with col2:
        avg_risk = ew_data['data']['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}",
            delta=f"{'High' if avg_risk > 0.6 else 'Medium' if avg_risk > 0.3 else 'Low'} Risk"
        )
    
    with col3:
        st.metric(
            label="Average Trust Score",
            value=f"{sm_data['summary']['average_trust_score']:.1f}/10",
            delta=f"{sm_data['summary']['trust_range']['min']:.1f}-{sm_data['summary']['trust_range']['max']:.1f}"
        )
    
    with col4:
        if gn_summary:
            st.metric(
                label="Good Neighbours Trust",
                value=f"{gn_summary['average_net_trust']:.2f}",
                delta=f"{gn_summary['net_trust_distribution']['positive_trust']} positive"
            )
        else:
            st.metric(
                label="Good Neighbours Trust",
                value="N/A",
                delta="Data not available"
            )
    
    with col5:
        st.metric(
            label="Intervention Cases",
            value=int_data['summary']['total_cases_in_database'],
            delta=f"{int_data['summary']['intervention_types_available']} types"
        )
    
    with col6:
        st.metric(
            label="Active Alerts",
            value=ew_data['summary']['total_alerts'],
            delta=f"{ew_data['summary']['anomalous_areas']} anomalies"
        )
    
    # Data Source Status
    st.subheader("📋 Data Source Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Available Data Sources:**
        - Good Neighbours Survey Data
        - Index of Multiple Deprivation (IMD)
        - Population Demographics (Census 2022)
        - Community Life Survey 2023-24
        - Early Warning System Indicators
        """)
    
    with col2:
        st.markdown("""
        **📊 Analysis Capabilities:**
        - Social Trust Analysis
        - Risk Assessment Mapping
        - Demographic Insights
        - Community Survey Analysis
        - Early Warning Indicators
        """)
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Social Trust Analysis", use_container_width=True):
            st.session_state.page = "Social Trust Analysis"
            st.rerun()
    
    with col2:
        if st.button("📋 Explore Community Survey", use_container_width=True):
            st.session_state.page = "Community Life Survey"
            st.rerun()
    
    with col3:
        if st.button("🚨 Early Warning System", use_container_width=True):
            st.session_state.page = "Early Warning System"
            st.rerun()
    
    # Interactive UK Map
    st.subheader("🗺️ UK Local Authority Districts")
    
    if FOLIUM_AVAILABLE and GEOPANDAS_AVAILABLE:
        with st.spinner("Loading interactive map..."):
            interactive_map, lads_data = create_interactive_uk_map()
            
        if interactive_map is not None:
            # Display the interactive map - made bigger
            map_data = st_folium(interactive_map, width=900, height=600, key="uk_map")
            
            # Check if a LAD was clicked on the map
            selected_lad_from_map = None
            
            # Method 1: Try GeoJSON object click detection
            if map_data and 'last_object_clicked' in map_data:
                clicked_data = map_data['last_object_clicked']
                if clicked_data and 'properties' in clicked_data:
                    properties = clicked_data['properties']
                    # Try to find the LAD name in the properties
                    for col in ['LAD24NM', 'LAD23NM', 'LAD22NM', 'LAD_NAME', 'Name', 'LADNM', 'LAD_NM', 'NAME']:
                        if col in properties:
                            selected_lad_from_map = properties[col]
                            break
            
            # Method 2: Try coordinate-based detection if Method 1 fails
            if not selected_lad_from_map and map_data and 'last_clicked' in map_data and lads_data is not None:
                clicked_coords = map_data['last_clicked']
                if clicked_coords:
                    selected_lad_from_map = find_lad_by_coordinates(clicked_coords, lads_data)
            
            # If a LAD was clicked on the map, store it in session state
            if selected_lad_from_map:
                st.session_state.selected_lad_from_map = selected_lad_from_map
                st.success(f"📍 Map Selection: {selected_lad_from_map}")
            
            # LAD Selection
            st.subheader("🔍 Select a Local Authority District")
            st.write("**Click on any LAD boundary on the map above, or choose from the dropdown below:**")
            
            # Get list of available LADs from the data
            available_lads = []
            if st.session_state.unified_data_connector.msoa_population_data is not None:
                available_lads = sorted(st.session_state.unified_data_connector.msoa_population_data['msoa_name'].unique().tolist())
            
            if available_lads:
                # Determine the default selection (from map click or previous selection)
                default_index = 0  # Default to empty selection
                if hasattr(st.session_state, 'selected_lad_from_map') and st.session_state.selected_lad_from_map in available_lads:
                    default_index = available_lads.index(st.session_state.selected_lad_from_map) + 1
                elif hasattr(st.session_state, 'selected_lad') and st.session_state.selected_lad in available_lads:
                    default_index = available_lads.index(st.session_state.selected_lad) + 1
                
                selected_lad_manual = st.selectbox(
                    "Select a Local Authority District:",
                    options=[""] + available_lads,
                    index=default_index,
                    key="manual_lad_selection"
                )
                
                if selected_lad_manual:
                    st.session_state.selected_lad = selected_lad_manual
                    st.success(f"📍 Selected: {selected_lad_manual}")
                    
                    # Get comprehensive data for the selected LAD
                    with st.spinner(f"Loading data for {selected_lad_manual}..."):
                        lad_data = get_lad_comprehensive_data(selected_lad_manual, st.session_state.unified_data_connector)
                    
                    # Display LAD data in tabs
                    st.subheader(f"📊 Data for {selected_lad_manual}")
                    
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "🏘️ Population", "📊 IMD Data", "🤝 Social Trust", "📋 Community Survey", "💼 Unemployment"])
                    
                    with tab1:
                        st.markdown(f"""
                        **Geographic Information:**
                        - **LAD Name:** {lad_data['lad_name']}
                        - **MSOA Count:** {lad_data['msoa_count']} (Middle Layer Super Output Areas)
                        - **Estimated LSOA Count:** {lad_data['lsoa_count']} (Lower Layer Super Output Areas)
                        """)
                        
                        if lad_data['msoa_count'] == 0:
                            st.warning("⚠️ No MSOA-level data available for this LAD")
                        else:
                            st.success(f"✅ Data available for {lad_data['msoa_count']} MSOAs")
                    
                    with tab2:
                        if lad_data['population_data'] is not None:
                            st.subheader("Population Demographics")
                            pop_data = lad_data['population_data']
                            
                            # Display key population metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_pop = pop_data['total_population'].sum()
                                st.metric("Total Population", f"{total_pop:,}")
                            with col2:
                                avg_pop_per_msoa = pop_data['total_population'].mean()
                                st.metric("Avg Population per MSOA", f"{avg_pop_per_msoa:,.0f}")
                            with col3:
                                st.metric("Number of MSOAs", len(pop_data))
                            
                            # Display detailed population data
                            st.subheader("Detailed Population Data by MSOA")
                            st.dataframe(pop_data, use_container_width=True)
                        else:
                            st.warning("⚠️ No population data available for this LAD")
                    
                    with tab3:
                        if lad_data['imd_data'] is not None:
                            st.subheader("Index of Multiple Deprivation (IMD) Data")
                            imd_data = lad_data['imd_data']
                            
                            # Display key IMD metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_decile = imd_data['msoa_imd_decile'].mean()
                                st.metric("Average IMD Decile", f"{avg_decile:.1f}")
                            with col2:
                                min_decile = imd_data['msoa_imd_decile'].min()
                                st.metric("Most Deprived MSOA", f"Decile {min_decile}")
                            with col3:
                                max_decile = imd_data['msoa_imd_decile'].max()
                                st.metric("Least Deprived MSOA", f"Decile {max_decile}")
                            
                            # Display detailed IMD data
                            st.subheader("Detailed IMD Data by MSOA")
                            st.dataframe(imd_data, use_container_width=True)
                        else:
                            st.warning("⚠️ No IMD data available for this LAD")
                    
                    with tab4:
                        if lad_data['good_neighbours_data'] is not None:
                            st.subheader("Good Neighbours Social Trust Data")
                            gn_data = lad_data['good_neighbours_data']
                            
                            # Display key trust metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_trust = gn_data['net_trust'].mean()
                                st.metric("Average Net Trust", f"{avg_trust:.2f}")
                            with col2:
                                positive_trust = (gn_data['net_trust'] > 0).sum()
                                st.metric("MSOA with Positive Trust", f"{positive_trust}/{len(gn_data)}")
                            with col3:
                                avg_always_trust = gn_data['always_usually_trust'].mean()
                                st.metric("Average 'Always/Usually Trust'", f"{avg_always_trust:.1f}%")
                            
                            # Display detailed trust data
                            st.subheader("Detailed Social Trust Data by MSOA")
                            st.dataframe(gn_data, use_container_width=True)
                        else:
                            st.warning("⚠️ No Good Neighbours social trust data available for this LAD")
                    
                    with tab5:
                        if lad_data['community_survey_data'] is not None:
                            st.subheader("Community Life Survey Data")
                            survey_data = lad_data['community_survey_data']
                            
                            # Display survey summary
                            st.markdown(f"""
                            **Survey Information:**
                            - **Total Responses:** {len(survey_data)}
                            - **Questions Covered:** {survey_data['question'].nunique()}
                            """)
                            
                            # Display detailed survey data
                            st.subheader("Detailed Community Life Survey Data")
                            st.dataframe(survey_data, use_container_width=True)
                        else:
                            st.warning("⚠️ No Community Life Survey data available for this LAD")
                    
                    with tab6:
                        if lad_data['unemployment_data'] is not None:
                            st.subheader("Unemployment Data")
                            unemployment = lad_data['unemployment_data']
                            
                            # Display unemployment metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Geography Code", unemployment.get('geography_code', 'N/A'))
                            with col2:
                                people_looking = unemployment.get('people_looking_for_work', 0)
                                st.metric("People Looking for Work", f"{people_looking:,}")
                            with col3:
                                unemployment_rate = unemployment.get('unemployment_proportion', 0)
                                st.metric("Unemployment Rate", f"{unemployment_rate:.1f}%")
                            
                            # Display detailed unemployment information
                            st.markdown(f"""
                            **Unemployment Details:**
                            - **Geography Name:** {unemployment.get('geography_name', 'N/A')}
                            - **Match Type:** {unemployment.get('match_type', 'N/A')}
                            - **People Looking for Work:** {unemployment.get('people_looking_for_work', 0):,}
                            - **Unemployment Proportion:** {unemployment.get('unemployment_proportion', 0):.2f}%
                            """)
                            
                            # Create a simple visualization
                            if unemployment.get('unemployment_proportion', 0) > 0:
                                import plotly.express as px
                                
                                # Create a simple bar chart
                                fig = px.bar(
                                    x=['Unemployment Rate'],
                                    y=[unemployment.get('unemployment_proportion', 0)],
                                    title=f"Unemployment Rate for {lad_data['lad_name']}",
                                    labels={'x': 'Metric', 'y': 'Percentage (%)'},
                                    color_discrete_sequence=['#ff6b6b']
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ No unemployment data available for this LAD")
            else:
                st.warning("⚠️ No LAD data available for manual selection")
            
            # Map information
            st.info("""
            **Interactive Map Features:**
            - Click and drag to pan around the UK
            - Use mouse wheel to zoom in/out
            - **Click on any LAD boundary to select it and populate the dropdown below**
            - Hover over LAD boundaries to see names and risk levels
            - Use layer control to toggle map layers
            - **Selected LAD will automatically populate the dropdown and show detailed data**
            """)
        else:
            st.error("Failed to create interactive map. Please check the data files.")
    else:
        st.warning("Interactive mapping requires Folium and GeoPandas. Please install: pip install folium geopandas streamlit-folium")

def early_warning_page():
    """Early warning system page"""
    st.header("🚨 Early Warning System")
    
    # Run early warning analysis
    with st.spinner("Running early warning analysis..."):
        ew_data = st.session_state.early_warning_system.run_full_analysis()
    
    # Key metrics
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Areas Monitored",
            value=ew_data['summary']['total_areas'],
            delta=f"{ew_data['summary']['critical_risk_areas']} critical"
        )
    
    with col2:
        avg_risk = ew_data['data']['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}",
            delta=f"{'High' if avg_risk > 0.6 else 'Medium' if avg_risk > 0.3 else 'Low'} Risk"
        )
    
    with col3:
        st.metric(
            label="Anomalous Areas",
            value=ew_data['summary']['anomalous_areas'],
            delta=f"{ew_data['summary']['total_alerts']} alerts"
        )
    
    with col4:
        st.metric(
            label="High Risk Areas",
            value=ew_data['summary']['high_risk_areas'],
            delta=f"{ew_data['summary']['critical_risk_areas']} critical"
        )
    
    # Risk level distribution
    st.subheader("📈 Risk Level Distribution")
    
    risk_dist = ew_data['data']['risk_level'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of risk levels
        fig_pie = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="Risk Level Distribution",
            color_discrete_map={
                'Low': '#2E8B57',
                'Medium': '#FFD700', 
                'High': '#FF8C00',
                'Critical': '#DC143C'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart of risk levels
        fig_bar = px.bar(
            x=risk_dist.index,
            y=risk_dist.values,
            title="Risk Level Counts",
            color=risk_dist.index,
            color_discrete_map={
                'Low': '#2E8B57',
                'Medium': '#FFD700',
                'High': '#FF8C00', 
                'Critical': '#DC143C'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # High-risk areas table
    st.subheader("🚨 High-Risk Areas")
    
    high_risk_data = ew_data['data'][ew_data['data']['risk_level'].isin(['High', 'Critical'])]
    high_risk_data = high_risk_data.sort_values('risk_score', ascending=False)
    
    if not high_risk_data.empty:
        # Display key columns
        display_cols = ['msoa_code', 'msoa_name', 'risk_score', 'risk_level', 'anomaly_score']
        if 'unemployment_rate' in high_risk_data.columns:
            display_cols.append('unemployment_rate')
        if 'crime_rate' in high_risk_data.columns:
            display_cols.append('crime_rate')
        if 'social_trust_score' in high_risk_data.columns:
            display_cols.append('social_trust_score')
        
        available_cols = [col for col in display_cols if col in high_risk_data.columns]
        
        st.dataframe(
            high_risk_data[available_cols],
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = high_risk_data[available_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download High-Risk Areas Data",
            data=csv,
            file_name=f"high_risk_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ No high-risk areas detected")
    
    # Anomaly detection results
    st.subheader("🔍 Anomaly Detection")
    
    anomalous_data = ew_data['data'][ew_data['data']['is_anomaly'] == True]
    
    if not anomalous_data.empty:
        st.warning(f"⚠️ {len(anomalous_data)} anomalous areas detected")
        
        # Anomaly scatter plot
        fig_scatter = px.scatter(
            anomalous_data,
            x='risk_score',
            y='anomaly_score',
            color='risk_level',
            hover_data=['msoa_code', 'msoa_name'],
            title="Anomalous Areas: Risk Score vs Anomaly Score",
            color_discrete_map={
                'Low': '#2E8B57',
                'Medium': '#FFD700',
                'High': '#FF8C00',
                'Critical': '#DC143C'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Anomalous areas table
        st.subheader("📋 Anomalous Areas Details")
        anomaly_cols = ['msoa_code', 'msoa_name', 'risk_score', 'anomaly_score', 'risk_level']
        available_anomaly_cols = [col for col in anomaly_cols if col in anomalous_data.columns]
        
        st.dataframe(
            anomalous_data[available_anomaly_cols],
            use_container_width=True
        )
    else:
        st.success("✅ No anomalous areas detected")
    
    # Risk factors analysis
    st.subheader("📊 Risk Factors Analysis")
    
    # Get risk distribution from summary
    risk_distribution = ew_data['summary'].get('risk_distribution', {})
    
    if risk_distribution:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Level Distribution:**")
            for level, count in risk_distribution.items():
                st.write(f"• {level}: {count} areas")
        
        with col2:
            avg_risk = ew_data['summary'].get('average_risk_score', 0)
            st.markdown("**Risk Statistics:**")
            st.write(f"• Average Risk Score: {avg_risk:.3f}")
            st.write(f"• Total Areas Monitored: {ew_data['summary']['total_areas']}")
            st.write(f"• Total Alerts Generated: {ew_data['summary']['total_alerts']}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    # Generate recommendations based on the data
    recommendations = []
    
    if ew_data['summary']['critical_risk_areas'] > 0:
        recommendations.append("Immediate intervention required for critical risk areas")
    
    if ew_data['summary']['anomalous_areas'] > 0:
        recommendations.append("Investigate anomalous areas for emerging social tensions")
    
    if ew_data['summary']['high_risk_areas'] > 5:
        recommendations.append("Consider broad-based community engagement programs")
    
    if ew_data['summary']['average_risk_score'] > 0.6:
        recommendations.append("Implement early warning monitoring across all areas")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.info("No specific recommendations available at this time")
    
    # Alert system status
    st.subheader("📧 Alert System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Alerts", ew_data['summary']['total_alerts'])
    
    with col2:
        st.metric("Email Alerts", "Configured" if st.session_state.alert_system.email_username else "Not Configured")
    
    with col3:
        st.metric("SMS Alerts", "Configured" if st.session_state.alert_system.twilio_account_sid else "Not Configured")
    
    # System settings
    with st.expander("⚙️ System Settings"):
        st.markdown("**Risk Thresholds:**")
        st.write(f"• Critical Risk: {st.session_state.early_warning_system.risk_threshold}")
        st.write(f"• Anomaly Detection: {st.session_state.early_warning_system.anomaly_detector.contamination}")
        
        st.markdown("**Data Sources:**")
        st.write("• IMD (Index of Multiple Deprivation)")
        st.write("• Good Neighbours Survey")
        st.write("• Population Demographics")
        st.write("• Community Life Survey")
        
        if st.button("🔄 Refresh Analysis"):
            st.rerun()

def sentiment_mapping_page():
    st.header("🗺️ Sentiment & Trust Mapping")
    st.info("Sentiment Mapping page - to be implemented")

def good_neighbours_page():
    st.header("🤝 Good Neighbours Trust Data")
    st.info("Good Neighbours page - to be implemented")

def community_survey_page():
    st.header("📋 Community Life Survey")
    st.info("Community Survey page - to be implemented")

def intervention_page():
    st.header("💡 Intervention Recommendations")
    st.info("Intervention page - to be implemented")

def engagement_simulator_page():
    st.header("🎯 Engagement Simulator")
    st.info("Engagement Simulator page - to be implemented")

def alert_management_page():
    st.header("📧 Alert Management")
    st.info("Alert Management page - to be implemented")

def genai_text_analysis_page():
    st.header("🤖 GenAI Text Analysis")
    st.info("GenAI Text Analysis page - to be implemented")

def settings_page():
    st.header("⚙️ System Settings")
    st.info("Settings page - to be implemented")

    # with col2:
    #     if analysis_type == "Area Profile":
    #         msoa_code = st.text_input("MSOA Code:", value="E02000001")
    #     else:
    #         n_areas = st.slider("Number of Areas:", 10, 200, 100)
    
    # Run analysis
    if st.button("Run Analysis"):
        with st.spinner("Running analysis..."):
            if analysis_type == "Full Analysis":
                results = st.session_state.early_warning_system.run_full_analysis()
                
                # Display results
                st.subheader("Analysis Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Areas", results['summary']['total_areas'])
                with col2:
                    st.metric("Critical Risk", results['summary']['critical_risk_areas'])
                with col3:
                    st.metric("Anomalies", results['summary']['anomalous_areas'])
                
                # Risk distribution
                st.subheader("Risk Distribution")
                risk_dist = results['data']['risk_level'].value_counts()
                fig = px.bar(x=risk_dist.index, y=risk_dist.values, title="Areas by Risk Level")
                st.plotly_chart(fig, use_container_width=True)
                
                # Alerts
                st.subheader("Generated Alerts")
                for alert in results['alerts']:
                    st.warning(f"**{alert['type']}** - {alert['message']}")
            
            elif analysis_type == "Risk Assessment":
                # Run risk assessment analysis
                data = st.session_state.early_warning_system.load_data(n_areas)
                data_with_risk = st.session_state.early_warning_system.calculate_risk_score(data)
                
                st.subheader("Risk Assessment Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Areas Analyzed", len(data_with_risk))
                with col2:
                    st.metric("Average Risk Score", f"{data_with_risk['risk_score'].mean():.3f}")
                with col3:
                    st.metric("High Risk Areas", len(data_with_risk[data_with_risk['risk_level'].isin(['High', 'Critical'])]))
                
                # Risk distribution chart
                st.subheader("Risk Level Distribution")
                risk_dist = data_with_risk['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    color_discrete_map={
                        'Low': '#4caf50',
                        'Medium': '#ff9800', 
                        'High': '#ff5722',
                        'Critical': '#f44336'
                    },
                    title="Distribution of Risk Levels"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk score distribution
                st.subheader("Risk Score Distribution")
                fig = px.histogram(
                    data_with_risk,
                    x='risk_score',
                    nbins=20,
                    title="Risk Score Distribution",
                    labels={'risk_score': 'Risk Score', 'count': 'Number of Areas'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top risk areas table
                st.subheader("Top Risk Areas")
                top_risk = data_with_risk.nlargest(10, 'risk_score')[['msoa_code', 'local_authority', 'risk_score', 'risk_level']]
                st.dataframe(top_risk, use_container_width=True)
                
                # Risk factors correlation
                st.subheader("Risk Factor Correlations")
                risk_factors = ['unemployment_rate', 'crime_rate', 'social_trust_score', 
                               'community_cohesion', 'economic_uncertainty', 'housing_stress']
                corr_data = data_with_risk[risk_factors + ['risk_score']].corr()
                
                fig = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix: Risk Factors vs Risk Score",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Anomaly Detection":
                # Run anomaly detection analysis
                data = st.session_state.early_warning_system.load_data(n_areas)
                data_with_risk = st.session_state.early_warning_system.calculate_risk_score(data)
                data_with_anomalies = st.session_state.early_warning_system.detect_anomalies(data_with_risk)
                
                st.subheader("Anomaly Detection Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Areas", len(data_with_anomalies))
                with col2:
                    st.metric("Anomalies Detected", len(data_with_anomalies[data_with_anomalies['is_anomaly'] == True]))
                with col3:
                    st.metric("Anomaly Rate", f"{len(data_with_anomalies[data_with_anomalies['is_anomaly'] == True]) / len(data_with_anomalies) * 100:.1f}%")
                
                # Anomaly score distribution
                st.subheader("Anomaly Score Distribution")
                fig = px.histogram(
                    data_with_anomalies,
                    x='anomaly_score',
                    color='is_anomaly',
                    nbins=20,
                    title="Anomaly Score Distribution",
                    labels={'anomaly_score': 'Anomaly Score', 'count': 'Number of Areas'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomalous areas table
                st.subheader("Anomalous Areas")
                anomalous_areas = data_with_anomalies[data_with_anomalies['is_anomaly'] == True][
                    ['msoa_code', 'local_authority', 'anomaly_score', 'risk_score', 'risk_level']
                ].sort_values('anomaly_score', ascending=True)
                
                if len(anomalous_areas) > 0:
                    st.dataframe(anomalous_areas, use_container_width=True)
                else:
                    st.info("No anomalous areas detected")
            
            elif analysis_type == "Area Profile":
                results = st.session_state.early_warning_system.run_full_analysis()
                area_data = results['data']
                profile = st.session_state.early_warning_system.get_risk_factors(area_data, msoa_code)
                
                if 'error' not in profile:
                    st.subheader(f"Area Profile: {msoa_code}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Risk Score", f"{profile['overall_risk_score']:.2f}")
                        st.metric("Risk Level", profile['risk_level'])
                    
                    with col2:
                        st.write("**Top Risk Factors:**")
                        for factor, score in profile['top_risk_factors']:
                            st.write(f"- {factor}: {score:.2f}")
                    
                    st.write("**Recommendations:**")
                    for rec in profile['recommendations']:
                        st.write(f"- {rec}")
                else:
                    st.error(f"Error: {profile['error']}")

def sentiment_mapping_page():
    """Sentiment mapping page"""
    st.header("🗺️ Sentiment & Trust Mapping")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        map_type = st.selectbox(
            "Map Type:",
            ["Trust Map", "Cohesion Map", "Sentiment Map", "Correlation Analysis", "Cohesion Dashboard"],
            key="sentiment_map_type_selectbox"
        )
    
    with col2:
        n_areas = st.slider("Number of Areas:", 20, 100, 50)
    
    if st.button("Generate Map"):
        with st.spinner("Generating map..."):
            results = st.session_state.sentiment_mapping.run_full_analysis()
            
            if map_type == "Trust Map":
                st.subheader("Social Trust Map")
                
                try:
                    # Try to use alternative map first (more reliable)
                    if 'alternative_map' in results['visualizations'] and results['visualizations']['alternative_map'] is not None:
                        st.plotly_chart(results['visualizations']['alternative_map'], use_container_width=True)
                    elif FOLIUM_AVAILABLE:
                        # Fallback to Folium map
                        trust_map = results['visualizations']['trust_map']
                        st_folium(trust_map, width=700, height=500)
                    else:
                        st.info("Interactive map not available. Install streamlit-folium for interactive maps.")
                        # Display map as static image or alternative visualization
                        st.plotly_chart(px.scatter(
                            results['data'],
                            x='longitude',
                            y='latitude',
                            color='social_trust_composite',
                            hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                            title="Social Trust Map (Static View)",
                            color_continuous_scale='RdYlGn'
                        ), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying map: {e}")
                    st.info("Displaying alternative visualization...")
                    # Fallback to scatter plot
                    st.plotly_chart(px.scatter(
                        results['data'],
                        x='longitude',
                        y='latitude',
                        color='social_trust_composite',
                        hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                        title="Social Trust Map (Alternative View)",
                        color_continuous_scale='RdYlGn'
                    ), use_container_width=True)
                
                # Trust statistics
                st.subheader("Trust Statistics")
                trust_data = results['data']['social_trust_composite']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Trust", f"{trust_data.mean():.1f}/10")
                with col2:
                    st.metric("Highest Trust", f"{trust_data.max():.1f}/10")
                with col3:
                    st.metric("Lowest Trust", f"{trust_data.min():.1f}/10")
            
            elif map_type == "Cohesion Map":
                st.subheader("Community Cohesion Map")
                
                try:
                    # Create cohesion map using Plotly scatter_mapbox
                    cohesion_fig = px.scatter_mapbox(
                        results['data'],
                        lat='latitude',
                        lon='longitude',
                        color='community_cohesion_composite',
                        hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                        color_continuous_scale='RdYlGn',
                        mapbox_style='carto-positron',
                        title='Community Cohesion Map',
                        zoom=10,
                        center=dict(lat=51.5074, lon=-0.1278)
                    )
                    
                    cohesion_fig.update_layout(
                        height=500,
                        margin=dict(r=0, t=30, l=0, b=0)
                    )
                    
                    st.plotly_chart(cohesion_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error displaying cohesion map: {e}")
                    st.info("Displaying alternative visualization...")
                    # Fallback to scatter plot
                    st.plotly_chart(px.scatter(
                        results['data'],
                        x='longitude',
                        y='latitude',
                        color='community_cohesion_composite',
                        hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                        title="Community Cohesion Map (Alternative View)",
                        color_continuous_scale='RdYlGn'
                    ), use_container_width=True)
                
                # Cohesion statistics
                st.subheader("Cohesion Statistics")
                cohesion_data = results['data']['community_cohesion_composite']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Cohesion", f"{cohesion_data.mean():.1f}/10")
                with col2:
                    st.metric("Highest Cohesion", f"{cohesion_data.max():.1f}/10")
                with col3:
                    st.metric("Lowest Cohesion", f"{cohesion_data.min():.1f}/10")
                
                # Cohesion components breakdown
                st.subheader("Cohesion Components")
                cohesion_components = results['data'][[
                    'community_belonging', 'volunteer_participation', 
                    'community_events_attendance', 'local_friendships'
                ]].mean()
                
                fig = px.bar(
                    x=cohesion_components.index,
                    y=cohesion_components.values,
                    title="Average Cohesion Component Scores",
                    labels={'x': 'Component', 'y': 'Score (1-10)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif map_type == "Sentiment Map":
                st.subheader("Sentiment Map")
                
                try:
                    # Create sentiment map using Plotly scatter_mapbox
                    sentiment_fig = px.scatter_mapbox(
                        results['data'],
                        lat='latitude',
                        lon='longitude',
                        color='sentiment_composite',
                        hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                        color_continuous_scale='RdYlGn',
                        mapbox_style='carto-positron',
                        title='Sentiment Map',
                        zoom=10,
                        center=dict(lat=51.5074, lon=-0.1278)
                    )
                    
                    sentiment_fig.update_layout(
                        height=500,
                        margin=dict(r=0, t=30, l=0, b=0)
                    )
                    
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error displaying sentiment map: {e}")
                    st.info("Displaying alternative visualization...")
                    # Fallback to scatter plot
                    st.plotly_chart(px.scatter(
                        results['data'],
                        x='longitude',
                        y='latitude',
                        color='sentiment_composite',
                        hover_data=['msoa_name', 'msoa_code', 'local_authority'],
                        title="Sentiment Map (Alternative View)",
                        color_continuous_scale='RdYlGn'
                    ), use_container_width=True)
                
                # Sentiment statistics
                st.subheader("Sentiment Statistics")
                sentiment_data = results['data']['sentiment_composite']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Sentiment", f"{sentiment_data.mean():.1f}/10")
                with col2:
                    st.metric("Highest Sentiment", f"{sentiment_data.max():.1f}/10")
                with col3:
                    st.metric("Lowest Sentiment", f"{sentiment_data.min():.1f}/10")
                
                # Sentiment components breakdown
                st.subheader("Sentiment Components")
                sentiment_components = results['data'][[
                    'overall_satisfaction', 'economic_optimism', 'future_outlook'
                ]].mean()
                
                fig = px.bar(
                    x=sentiment_components.index,
                    y=sentiment_components.values,
                    title="Average Sentiment Component Scores",
                    labels={'x': 'Component', 'y': 'Score (1-10)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif map_type == "Cohesion Dashboard":
                st.subheader("Cohesion Dashboard")
                
                # Display the cohesion dashboard visualization
                cohesion_dashboard_fig = results['visualizations']['cohesion_dashboard']
                st.plotly_chart(cohesion_dashboard_fig, use_container_width=True)
                
                # Additional cohesion insights
                st.subheader("Cohesion Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Cohesion Areas")
                    top_cohesion = results['data'].nlargest(5, 'community_cohesion_composite')[
                        ['msoa_name', 'msoa_code', 'local_authority', 'community_cohesion_composite']
                    ]
                    
                    for idx, row in top_cohesion.iterrows():
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #4caf50; margin-bottom: 0.5rem;">
                            <strong>{row['msoa_name']}</strong> - {row['local_authority']}<br>
                            Cohesion Score: {row['community_cohesion_composite']:.1f}/10
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Cohesion vs Trust Relationship")
                    fig = px.scatter(
                        results['data'],
                        x='social_trust_composite',
                        y='community_cohesion_composite',
                        color='local_authority',
                        hover_data=['msoa_name', 'msoa_code'],
                        title="Community Cohesion vs Social Trust",
                        labels={
                            'social_trust_composite': 'Social Trust Score',
                            'community_cohesion_composite': 'Community Cohesion Score'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Overall cohesion metrics
                st.subheader("Overall Cohesion Metrics")
                overall_cohesion = results['data']['overall_cohesion_score']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Overall Cohesion", f"{overall_cohesion.mean():.1f}/10")
                with col2:
                    st.metric("Social Trust Component", f"{results['data']['social_trust_composite'].mean():.1f}/10")
                with col3:
                    st.metric("Community Cohesion Component", f"{results['data']['community_cohesion_composite'].mean():.1f}/10")
                with col4:
                    st.metric("Sentiment Component", f"{results['data']['sentiment_composite'].mean():.1f}/10")
            
            elif map_type == "Correlation Analysis":
                st.subheader("Correlation Analysis")
                
                # Correlation heatmap
                corr_fig = results['visualizations']['correlation_heatmap']
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Scatter plot
                scatter_fig = results['visualizations']['deprivation_scatter']
                st.plotly_chart(scatter_fig, use_container_width=True)

def good_neighbours_page():
    """Good Neighbours social trust data page"""
    st.header("🤝 Good Neighbours Social Trust Data")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Overview", "MSOA Lookup", "Top Trust Areas", "Lowest Trust Areas", "Trust Distribution"],
            key="good_neighbours_analysis_selectbox"
        )
    
    with col2:
        if analysis_type == "MSOA Lookup":
            msoa_code = st.text_input("MSOA Code:", value="E02000001")
        elif analysis_type in ["Top Trust Areas", "Lowest Trust Areas"]:
            n_areas = st.slider("Number of Areas:", 5, 50, 10)
    
    # Run analysis
    if st.button("Run Analysis"):
        with st.spinner("Loading Good Neighbours data..."):
            connector = st.session_state.unified_data_connector
            
            if analysis_type == "Overview":
                # Load data and show summary
                if connector.good_neighbours_data is not None:
                    df = connector.good_neighbours_data
                    summary = connector.get_good_neighbours_summary()
                    
                    if df is not None and summary is not None:
                        st.subheader("📊 Data Overview")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total MSOAs", summary['total_msoas'])
                        
                        with col2:
                            avg_trust = df['net_trust'].mean()
                            st.metric("Average Net Trust", f"{avg_trust:.2f}")
                        
                        with col3:
                            max_trust = df['net_trust'].max()
                            st.metric("Highest Trust", f"{max_trust:.2f}")
                        
                        with col4:
                            min_trust = df['net_trust'].min()
                            st.metric("Lowest Trust", f"{min_trust:.2f}")
                    
                        # Trust distribution
                        st.subheader("Trust Distribution")
                        
                        # Create trust distribution chart
                        fig = px.histogram(df, x='net_trust', nbins=20, title='Net Trust Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trust categories
                        positive_trust = len(df[df['net_trust'] > 0])
                        negative_trust = len(df[df['net_trust'] < 0])
                        neutral_trust = len(df[df['net_trust'] == 0])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive Trust", positive_trust)
                        with col2:
                            st.metric("Negative Trust", negative_trust)
                        with col3:
                            st.metric("Neutral Trust", neutral_trust)
                else:
                    st.error("Failed to load Good Neighbours data")
            
            elif analysis_type == "MSOA Lookup":
                results = connector.get_msoa_data(msoa_code, ['good_neighbours'])
                trust_result = results.get('good_neighbours')
                
                if trust_result and trust_result.success:
                    trust_data = trust_result.data
                    st.subheader(f"Social Trust Data for {msoa_code}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("MSOA Name", trust_data['msoa_name'])
                        st.metric("Net Trust Score", f"{trust_data['net_trust']:.2f}")
                    
                    with col2:
                        st.metric("Always/Usually Trust", f"{trust_data['always_usually_trust']:.1f}%")
                        st.metric("Usually/Almost Always Careful", f"{trust_data['usually_almost_always_careful']:.1f}%")
                    
                    # Trust interpretation
                    if trust_data['net_trust'] > 0:
                        st.success(f"✅ Positive net trust score indicates higher trust than caution")
                    elif trust_data['net_trust'] < 0:
                        st.warning(f"⚠️ Negative net trust score indicates higher caution than trust")
                    else:
                        st.info(f"ℹ️ Neutral net trust score indicates balanced trust and caution")
                        
                else:
                    st.error(f"No social trust data found for MSOA {msoa_code}")
            
            elif analysis_type == "Top Trust Areas":
                top_areas = connector.get_top_performing_msoas('net_trust', n_areas, 'good_neighbours')
                
                if top_areas:
                    st.subheader(f"Top {n_areas} Trust Areas")
                    
                    # Create DataFrame for display
                    top_df = pd.DataFrame(top_areas)
                    
                    # Display as table
                    st.dataframe(
                        top_df[['msoa_name', 'msoa_code', 'net_trust', 'always_usually_trust', 'usually_almost_always_careful']],
                        use_container_width=True
                    )
                    
                    # Bar chart
                    fig = px.bar(
                        top_df,
                        x='msoa_name',
                        y='net_trust',
                        title=f"Top {n_areas} MSOAs by Net Trust Score",
                        labels={'net_trust': 'Net Trust Score', 'msoa_name': 'MSOA Name'},
                        color='net_trust',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to load top trust areas")
            
            elif analysis_type == "Lowest Trust Areas":
                # Get lowest by reversing the order (ascending instead of descending)
                all_areas = connector.get_top_performing_msoas('net_trust', 1000, 'good_neighbours')
                lowest_areas = sorted(all_areas, key=lambda x: x['net_trust'])[:n_areas]
                
                if lowest_areas:
                    st.subheader(f"Lowest {n_areas} Trust Areas")
                    
                    # Create DataFrame for display
                    lowest_df = pd.DataFrame(lowest_areas)
                    
                    # Display as table
                    st.dataframe(
                        lowest_df[['msoa_name', 'msoa_code', 'net_trust', 'always_usually_trust', 'usually_almost_always_careful']],
                        use_container_width=True
                    )
                    
                    # Bar chart
                    fig = px.bar(
                        lowest_df,
                        x='msoa_name',
                        y='net_trust',
                        title=f"Lowest {n_areas} MSOAs by Net Trust Score",
                        labels={'net_trust': 'Net Trust Score', 'msoa_name': 'MSOA Name'},
                        color='net_trust',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to load lowest trust areas")
            
            elif analysis_type == "Trust Distribution":
                df = connector.load_good_neighbours_data()
                summary = connector.get_good_neighbours_summary()
                
                if df is not None and summary is not None:
                    st.subheader("Trust Score Distribution Analysis")
                    
                    # Decile analysis
                    df['trust_decile'] = pd.qcut(df['net_trust'], q=10, labels=False, duplicates='drop') + 1
                    
                    decile_stats = df.groupby('trust_decile')['net_trust'].agg(['count', 'mean', 'min', 'max']).reset_index()
                    decile_stats.columns = ['Decile', 'Count', 'Mean Trust', 'Min Trust', 'Max Trust']
                    
                    st.subheader("Trust Score Deciles")
                    st.dataframe(decile_stats, use_container_width=True)
                    
                    # Decile distribution chart
                    fig = px.bar(
                        decile_stats,
                        x='Decile',
                        y='Mean Trust',
                        title="Average Trust Score by Decile",
                        labels={'Mean Trust': 'Average Net Trust Score', 'Decile': 'Trust Decile'},
                        color='Mean Trust',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trust vs caution scatter
                    fig = px.scatter(
                        df,
                        x='always_usually_trust',
                        y='usually_almost_always_careful',
                        color='net_trust',
                        hover_data=['msoa_name', 'msoa_code'],
                        title="Trust vs Caution Relationship",
                        color_continuous_scale='RdYlGn',
                        labels={
                            'always_usually_trust': 'Always/Usually Trust (%)',
                            'usually_almost_always_careful': 'Usually/Almost Always Careful (%)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to load trust distribution data")

def intervention_page():
    """Intervention recommendations page"""
    st.header("💡 Intervention Recommendations")
    
    # Area input
    st.subheader("Target Area Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population = st.number_input("Population:", value=8500, min_value=1000, max_value=50000)
        unemployment_rate = st.slider("Unemployment Rate (%):", 0.0, 30.0, 8.5)
        crime_rate = st.slider("Crime Rate:", 0.0, 200.0, 75.0)
        income_deprivation = st.slider("Income Deprivation (%):", 0.0, 100.0, 25.0)
    
    with col2:
        education_attainment = st.slider("Education Attainment (%):", 0.0, 100.0, 68.0)
        ethnic_diversity = st.slider("Ethnic Diversity (%):", 0.0, 100.0, 35.0)
        age_dependency_ratio = st.slider("Age Dependency Ratio:", 0.0, 2.0, 0.65)
        housing_stress = st.slider("Housing Stress (%):", 0.0, 100.0, 40.0)
    
    # Create target area
    target_area = {
        'population': population,
        'unemployment_rate': unemployment_rate,
        'crime_rate': crime_rate,
        'income_deprivation': income_deprivation,
        'education_attainment': education_attainment,
        'ethnic_diversity': ethnic_diversity,
        'age_dependency_ratio': age_dependency_ratio,
        'housing_stress': housing_stress,
        'social_trust_pre': 5.8,
        'community_cohesion_pre': 6.1,
        'volunteer_rate_pre': 15
    }
    
    if st.button("Get Recommendations"):
        with st.spinner("Analyzing similar cases..."):
            results = st.session_state.intervention_tool.run_full_analysis(target_area)
            
            st.subheader("Recommended Interventions")
            
            for i, rec in enumerate(results['recommendations'], 1):
                with st.expander(f"{i}. {rec['intervention_type']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Expected Success Score:** {rec['expected_success_score']:.2f}")
                        st.write(f"**Trust Improvement:** {rec['expected_trust_improvement']:.2f}")
                        st.write(f"**Cohesion Improvement:** {rec['expected_cohesion_improvement']:.2f}")
                        st.write(f"**Volunteer Improvement:** {rec['expected_volunteer_improvement']:.2f}")
                    
                    with col2:
                        st.write(f"**Average Funding:** £{rec['average_funding_required']:,.0f}")
                        st.write(f"**Duration:** {rec['average_duration_months']:.0f} months")
                        st.write(f"**Cost Effectiveness:** {rec['cost_effectiveness']:.3f}")
                        st.write(f"**Confidence:** {rec['confidence_level']:.1%}")
                    
                    st.write("**Key Components:**")
                    for component in rec['key_components']:
                        st.write(f"- {component}")
                    
                    st.write("**Success Factors:**")
                    for factor in rec['success_factors']:
                        st.write(f"- {factor}")

def engagement_simulator_page():
    """Engagement simulator page"""
    st.header("🎯 Community Engagement Simulator")
    
    # Scenario selection
    scenario_type = st.selectbox(
        "Scenario Type:",
        ["Custom Scenario", "Predefined Scenarios", "Optimization"],
        key="engagement_scenario_type_selectbox"
    )
    
    if scenario_type == "Custom Scenario":
        st.subheader("Custom Intervention Scenario")
        
        # Intervention sliders
        interventions = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            interventions['community_events_investment'] = st.slider("Community Events", 0, 100, 50)
            interventions['youth_programs_investment'] = st.slider("Youth Programs", 0, 100, 30)
            interventions['interfaith_dialogue_investment'] = st.slider("Interfaith Dialogue", 0, 100, 20)
            interventions['neighborhood_watch_investment'] = st.slider("Neighborhood Watch", 0, 100, 40)
            interventions['community_garden_investment'] = st.slider("Community Garden", 0, 100, 25)
            interventions['skills_training_investment'] = st.slider("Skills Training", 0, 100, 35)
        
        with col2:
            interventions['cultural_exchange_investment'] = st.slider("Cultural Exchange", 0, 100, 15)
            interventions['mental_health_support_investment'] = st.slider("Mental Health Support", 0, 100, 45)
            interventions['digital_inclusion_investment'] = st.slider("Digital Inclusion", 0, 100, 20)
            interventions['sports_recreation_investment'] = st.slider("Sports & Recreation", 0, 100, 30)
            interventions['volunteer_coordination_investment'] = st.slider("Volunteer Coordination", 0, 100, 25)
            interventions['local_business_support_investment'] = st.slider("Local Business Support", 0, 100, 10)
        
        # Baseline area (simplified)
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
        
        if st.button("Simulate Impact"):
            with st.spinner("Running simulation..."):
                result = st.session_state.engagement_simulator.simulate_intervention_impact(
                    baseline_area, interventions
                )
                
                # Display results
                st.subheader("Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trust Improvement", f"{result['improvements']['trust_improvement']:.2f}")
                    st.metric("Cohesion Improvement", f"{result['improvements']['cohesion_improvement']:.2f}")
                
                with col2:
                    st.metric("Sentiment Improvement", f"{result['improvements']['sentiment_improvement']:.2f}")
                    st.metric("Overall Improvement", f"{result['improvements']['overall_improvement']:.2f}")
                
                with col3:
                    st.metric("Total Investment", f"£{result['investment_summary']['total_investment']:,.0f}")
                    st.metric("Cost Effectiveness", f"{result['investment_summary']['cost_effectiveness']:.3f}")
                
                # Before/After comparison
                st.subheader("Before vs After")
                
                comparison_data = pd.DataFrame({
                    'Metric': ['Social Trust', 'Community Cohesion', 'Sentiment'],
                    'Before': [
                        baseline_area['social_trust_baseline'],
                        baseline_area['community_cohesion_baseline'],
                        baseline_area['sentiment_baseline']
                    ],
                    'After': [
                        result['predicted_outcomes']['social_trust_final'],
                        result['predicted_outcomes']['community_cohesion_final'],
                        result['predicted_outcomes']['sentiment_final']
                    ]
                })
                
                fig = px.bar(
                    comparison_data.melt(id_vars='Metric'),
                    x='Metric',
                    y='value',
                    color='variable',
                    title="Before vs After Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Optimization":
        st.subheader("Intervention Optimization")
        
        budget = st.slider("Available Budget:", 500, 5000, 1000)
        target_outcome = st.selectbox(
            "Target Outcome:",
            ["overall", "trust", "cohesion", "sentiment"],
            key="engagement_target_outcome_selectbox"
        )
        
        if st.button("Optimize Interventions"):
            with st.spinner("Optimizing intervention mix..."):
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
                
                result = st.session_state.engagement_simulator.optimize_intervention_mix(
                    baseline_area, budget, target_outcome
                )
                
                if result['optimization_successful']:
                    st.success("Optimization completed successfully!")
                    
                    # Display optimized interventions
                    st.subheader("Optimized Intervention Mix")
                    
                    interventions = result['optimized_interventions']
                    simulation = result['simulation_result']
                    
                    # Create intervention chart
                    intervention_df = pd.DataFrame(
                        list(interventions.items()),
                        columns=['Intervention', 'Investment']
                    )
                    intervention_df = intervention_df[intervention_df['Investment'] > 0]
                    
                    fig = px.bar(
                        intervention_df,
                        x='Investment',
                        y='Intervention',
                        orientation='h',
                        title="Optimized Investment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Expected Trust Improvement", f"{simulation['improvements']['trust_improvement']:.2f}")
                        st.metric("Expected Cohesion Improvement", f"{simulation['improvements']['cohesion_improvement']:.2f}")
                    
                    with col2:
                        st.metric("Expected Sentiment Improvement", f"{simulation['improvements']['sentiment_improvement']:.2f}")
                        st.metric("Expected Overall Improvement", f"{simulation['improvements']['overall_improvement']:.2f}")
                else:
                    st.error(f"Optimization failed: {result['error']}")

def alert_management_page():
    """Alert management page"""
    st.header("📧 Alert Management")
    
    # Alert system status
    st.subheader("Alert System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Email Configured", "✅" if st.session_state.alert_system.email_username else "❌")
    
    with col2:
        st.metric("SMS Configured", "✅" if st.session_state.alert_system.twilio_account_sid else "❌")
    
    with col3:
        st.metric("Alert Threshold", f"{st.session_state.alert_system.thresholds['critical_risk']:.1f}")
    
    # Test alerts
    st.subheader("Test Alert System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_email = st.text_input("Test Email Address:")
        if st.button("Send Test Email"):
            if test_email:
                test_result = st.session_state.alert_system.test_alert_system(test_email=test_email)
                if test_result['email_test']:
                    st.success("Test email sent successfully!")
                else:
                    st.error(f"Failed to send test email: {test_result['email_error']}")
            else:
                st.warning("Please enter an email address")
    
    with col2:
        test_phone = st.text_input("Test Phone Number:")
        if st.button("Send Test SMS"):
            if test_phone:
                test_result = st.session_state.alert_system.test_alert_system(test_phone=test_phone)
                if test_result['sms_test']:
                    st.success("Test SMS sent successfully!")
                else:
                    st.error(f"Failed to send test SMS: {test_result['sms_error']}")
            else:
                st.warning("Please enter a phone number")
    
    # Alert configuration
    st.subheader("Alert Configuration")
    
    with st.expander("Configure Alert Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            critical_threshold = st.slider("Critical Risk Threshold:", 0.0, 1.0, 0.8)
            high_threshold = st.slider("High Risk Threshold:", 0.0, 1.0, 0.6)
        
        with col2:
            medium_threshold = st.slider("Medium Risk Threshold:", 0.0, 1.0, 0.4)
            anomaly_threshold = st.slider("Anomaly Threshold:", -1.0, 1.0, -0.5)
        
        if st.button("Update Thresholds"):
            settings = {
                'thresholds': {
                    'critical_risk': critical_threshold,
                    'high_risk': high_threshold,
                    'medium_risk': medium_threshold,
                    'anomaly_threshold': anomaly_threshold
                }
            }
            st.session_state.alert_system.configure_alert_settings(settings)
            st.success("Alert thresholds updated!")

def settings_page():
    """System settings page"""
    st.header("⚙️ System Settings")
    
    # Data source configuration
    st.subheader("Data Source Configuration")
    
    with st.expander("Configure Data Sources"):
        st.write("**Available Data Sources:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Indices of Multiple Deprivation (IMD)", value=True)
            st.checkbox("ONS Census Data", value=True)
            st.checkbox("Community Life Survey", value=False)
        
        with col2:
            st.checkbox("Social Media Sentiment", value=False)
            st.checkbox("Crime Statistics", value=True)
            st.checkbox("Economic Indicators", value=True)
    
    # Model configuration
    st.subheader("Model Configuration")
    
    with st.expander("Configure ML Models"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Anomaly Detection Sensitivity:", 0.01, 0.5, 0.1)
            st.slider("Risk Score Weight - Trust:", 0.0, 1.0, 0.3)
            st.slider("Risk Score Weight - Cohesion:", 0.0, 1.0, 0.3)
        
        with col2:
            st.slider("Risk Score Weight - Sentiment:", 0.0, 1.0, 0.2)
            st.slider("Risk Score Weight - Deprivation:", 0.0, 1.0, 0.2)
            st.slider("Model Update Frequency (days):", 1, 30, 7)
    
    # Export/Import settings
    st.subheader("Data Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Settings"):
            settings = {
                'thresholds': st.session_state.alert_system.thresholds,
                'export_timestamp': datetime.now().isoformat()
            }
            st.download_button(
                label="Download Settings JSON",
                data=json.dumps(settings, indent=2),
                file_name=f"social_cohesion_settings_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Settings", type=['json'])
        if uploaded_file is not None:
            try:
                settings = json.load(uploaded_file)
                st.session_state.alert_system.configure_alert_settings(settings)
                st.success("Settings imported successfully!")
            except Exception as e:
                st.error(f"Error importing settings: {e}")

def genai_text_analysis_page():
    """GenAI Text Analysis page"""
    st.header("🤖 GenAI Text Analysis")
    st.markdown("Analyze text for social cohesion issues using Azure OpenAI")
    
    # Check if Azure OpenAI is configured
    try:
        analyzer = GenAITextAnalyzer()
        st.success("✅ Azure OpenAI connection configured")
    except Exception as e:
        st.error(f"❌ Azure OpenAI configuration error: {e}")
        st.info("Please configure Azure OpenAI settings in your .env file")
        return
    
    # Create tabs for different analysis modes
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Single Text Analysis", "📁 Batch Analysis", "🗺️ Locality Mapping", "🔍 Embedding & Similarity", "📊 Analysis Results"])
    
    with tab1:
        st.subheader("Single Text Analysis")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Upload file", "Paste from clipboard"]
        )
        
        text_content = ""
        
        if input_method == "Type text":
            text_content = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Enter survey responses, social media posts, reports, or any text related to social cohesion..."
            )
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'csv', 'json'],
                help="Supported formats: .txt, .csv, .json"
            )
            if uploaded_file is not None:
                try:
                    text_content = str(uploaded_file.read(), "utf-8")
                    st.success(f"File uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:  # Paste from clipboard
            text_content = st.text_area(
                "Paste text from clipboard:",
                height=200,
                placeholder="Paste text here..."
            )
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox(
                "Text source:",
                ["survey", "social_media", "report", "interview", "feedback", "other"],
                key="genai_text_source_selectbox"
            )
        with col2:
            analysis_type = st.selectbox(
                "Analysis focus:",
                ["comprehensive", "social_cohesion_only", "location_focused", "sentiment_only"],
                key="genai_analysis_focus_selectbox"
            )
        
        # Analyze button
        if st.button("🔍 Analyze Text", type="primary"):
            if not text_content.strip():
                st.warning("Please enter some text to analyze")
            else:
                with st.spinner("Analyzing text with GenAI..."):
                    try:
                        result = analyzer.analyze_text(text_content, source)
                        
                        # Display results
                        st.success("Analysis completed!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Issues", result.total_issues)
                        with col2:
                            st.metric("Critical Issues", result.critical_issues, delta=None)
                        with col3:
                            st.metric("High Priority", result.high_issues, delta=None)
                        with col4:
                            st.metric("Localities Found", len(result.localities_found))
                        
                        # Issues breakdown
                        if result.issues:
                            st.subheader("🚨 Issues Identified")
                            
                            # Create a DataFrame for better display
                            issues_data = []
                            for issue in result.issues:
                                issues_data.append({
                                    "Type": issue.issue_type.replace("_", " ").title(),
                                    "Severity": issue.severity,
                                    "Confidence": f"{issue.confidence:.2f}",
                                    "Description": issue.description,
                                    "Location": issue.location_mentioned or "Not specified",
                                    "MSOA": issue.msoa_code or "Not mapped",
                                    "Local Authority": issue.local_authority or "Not specified"
                                })
                            
                            issues_df = pd.DataFrame(issues_data)
                            st.dataframe(issues_df, use_container_width=True)
                            
                            # Severity distribution
                            severity_counts = issues_df['Severity'].value_counts()
                            fig = px.pie(
                                values=severity_counts.values,
                                names=severity_counts.index,
                                title="Issue Severity Distribution",
                                color_discrete_map={
                                    "Critical": "#FF0000",
                                    "High": "#FF8C00",
                                    "Medium": "#FFD700",
                                    "Low": "#90EE90"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Localities found
                        if result.localities_found:
                            st.subheader("🗺️ Localities Identified")
                            localities_data = []
                            for locality in result.localities_found:
                                localities_data.append({
                                    "Name": locality['name'],
                                    "Type": locality['type'],
                                    "MSOA Code": locality.get('msoa_code', 'Not mapped'),
                                    "Context": locality.get('context', '')
                                })
                            
                            localities_df = pd.DataFrame(localities_data)
                            st.dataframe(localities_df, use_container_width=True)
                        
                        # Recommendations
                        if result.recommendations:
                            st.subheader("💡 Recommendations")
                            for i, rec in enumerate(result.recommendations, 1):
                                st.write(f"{i}. {rec}")
                        
                        # Export options
                        st.subheader("📥 Export Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(analyzer._result_to_dict(result), indent=2),
                                file_name=f"genai_analysis_{result.text_id}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            csv_data = analyzer.export_results([result], 'csv')
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"genai_analysis_{result.text_id}.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            summary_data = analyzer.export_results([result], 'summary')
                            st.download_button(
                                label="Download Summary",
                                data=summary_data,
                                file_name=f"genai_analysis_{result.text_id}.txt",
                                mime="text/plain"
                            )
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.info("Please check your Azure OpenAI configuration")
    
    with tab2:
        st.subheader("Batch Analysis")
        st.info("Upload multiple texts for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Upload text files:",
            type=['txt', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple files for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("🔍 Analyze All Files", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    try:
                        texts = []
                        for file in uploaded_files:
                            content = str(file.read(), "utf-8")
                            texts.append((content, f"file_{file.name}", file.name))
                        
                        results = analyzer.analyze_multiple_texts(texts)
                        
                        # Summary statistics
                        total_issues = sum(result.total_issues for result in results)
                        total_critical = sum(result.critical_issues for result in results)
                        total_files = len(results)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Processed", total_files)
                        with col2:
                            st.metric("Total Issues", total_issues)
                        with col3:
                            st.metric("Critical Issues", total_critical)
                        
                        # Results table
                        results_data = []
                        for result in results:
                            results_data.append({
                                "File": result.text_id,
                                "Source": result.source,
                                "Total Issues": result.total_issues,
                                "Critical": result.critical_issues,
                                "High": result.high_issues,
                                "Medium": result.medium_issues,
                                "Low": result.low_issues,
                                "Localities": len(result.localities_found)
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Export batch results
                        st.subheader("📥 Export Batch Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            batch_json = analyzer.export_results(results, 'json')
                            st.download_button(
                                label="Download Batch JSON",
                                data=batch_json,
                                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            batch_csv = analyzer.export_results(results, 'csv')
                            st.download_button(
                                label="Download Batch CSV",
                                data=batch_csv,
                                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Batch analysis failed: {e}")
    
    with tab3:
        st.subheader("Locality Mapping")
        st.info("Map localities to MSOA codes")
        
        mapper = LocalityMapper()
        
        # Single locality mapping
        st.write("**Single Locality Mapping**")
        locality_input = st.text_input(
            "Enter locality name:",
            placeholder="e.g., Kensington, SW1A 1AA, Hyde Park"
        )
        
        if st.button("🗺️ Map Locality"):
            if locality_input:
                result = mapper.map_locality(locality_input)
                if result:
                    st.success("✅ Locality mapped successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {result.name}")
                        st.write(f"**Type:** {result.type}")
                        st.write(f"**MSOA Code:** {result.msoa_code}")
                    with col2:
                        st.write(f"**Local Authority:** {result.local_authority}")
                        st.write(f"**Region:** {result.region}")
                        st.write(f"**Confidence:** {result.confidence:.2f}")
                else:
                    st.warning("❌ No mapping found for this locality")
                    
                    # Try fuzzy search
                    search_results = mapper.search_localities(locality_input)
                    if search_results:
                        st.write("**Similar localities found:**")
                        for result in search_results[:5]:
                            st.write(f"- {result.name} ({result.type}) -> {result.msoa_code}")
        
        # Locality search
        st.write("**Locality Search**")
        search_query = st.text_input(
            "Search for localities:",
            placeholder="e.g., park, station, market"
        )
        
        if st.button("🔍 Search Localities"):
            if search_query:
                results = mapper.search_localities(search_query)
                if results:
                    st.write(f"Found {len(results)} localities:")
                    
                    search_data = []
                    for result in results:
                        search_data.append({
                            "Name": result.name,
                            "Type": result.type,
                            "MSOA Code": result.msoa_code,
                            "Local Authority": result.local_authority,
                            "Confidence": f"{result.confidence:.2f}"
                        })
                    
                    search_df = pd.DataFrame(search_data)
                    st.dataframe(search_df, use_container_width=True)
                else:
                    st.info("No localities found for this search")
        
        # MSOA validation
        st.write("**MSOA Code Validation**")
        msoa_input = st.text_input(
            "Enter MSOA code to validate:",
            placeholder="e.g., E02000001"
        )
        
        if st.button("✅ Validate MSOA"):
            if msoa_input:
                is_valid = mapper.validate_msoa_code(msoa_input)
                if is_valid:
                    msoa_info = mapper.get_msoa_info(msoa_input)
                    st.success(f"✅ MSOA code {msoa_input} is valid")
                    st.write(f"**Local Authority:** {msoa_info['la']}")
                    st.write(f"**Region:** {msoa_info['region']}")
                else:
                    st.error(f"❌ MSOA code {msoa_input} is not valid")
    
    with tab4:
        st.subheader("Embedding & Similarity Analysis")
        st.info("Generate embeddings and calculate text similarity using Azure OpenAI")
        
        # Embedding generation
        st.write("**Generate Embeddings**")
        embed_text = st.text_area(
            "Enter text to generate embedding:",
            height=100,
            placeholder="Enter text to convert to vector representation..."
        )
        
        if st.button("🔢 Generate Embedding"):
            if embed_text:
                with st.spinner("Generating embedding..."):
                    try:
                        embedding = analyzer.generate_single_embedding(embed_text)
                        
                        st.success("✅ Embedding generated successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Model", analyzer.embedding_model)
                        with col2:
                            st.metric("Dimensions", len(embedding))
                        
                        # Show embedding statistics
                        import numpy as np
                        embedding_array = np.array(embedding)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{embedding_array.mean():.4f}")
                        with col2:
                            st.metric("Std Dev", f"{embedding_array.std():.4f}")
                        with col3:
                            st.metric("Min", f"{embedding_array.min():.4f}")
                        
                        # Show first few dimensions
                        st.write("**First 20 dimensions:**")
                        st.code(embedding[:20])
                        
                        # Download embedding
                        embed_data = {
                            "text": embed_text,
                            "embedding": embedding,
                            "model": analyzer.embedding_model,
                            "dimensions": len(embedding),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.download_button(
                            label="📥 Download Embedding JSON",
                            data=json.dumps(embed_data, indent=2),
                            file_name=f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    except Exception as e:
                        st.error(f"Error generating embedding: {e}")
            else:
                st.warning("Please enter text to generate embedding")
        
        st.markdown("---")
        
        # Text similarity
        st.write("**Text Similarity Analysis**")
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "First text:",
                height=100,
                placeholder="Enter first text for comparison..."
            )
        
        with col2:
            text2 = st.text_area(
                "Second text:",
                height=100,
                placeholder="Enter second text for comparison..."
            )
        
        if st.button("🔍 Calculate Similarity"):
            if text1 and text2:
                with st.spinner("Calculating similarity..."):
                    try:
                        similarity_score = analyzer.calculate_text_similarity(text1, text2)
                        
                        st.success("✅ Similarity calculated!")
                        
                        # Display similarity score
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Similarity Score", f"{similarity_score:.4f}")
                        with col2:
                            if similarity_score > 0.8:
                                st.metric("Interpretation", "Very Similar", delta="✅")
                            elif similarity_score > 0.6:
                                st.metric("Interpretation", "Moderately Similar", delta="🟡")
                            elif similarity_score > 0.4:
                                st.metric("Interpretation", "Somewhat Similar", delta="🟠")
                            else:
                                st.metric("Interpretation", "Dissimilar", delta="❌")
                        with col3:
                            st.metric("Confidence", f"{similarity_score * 100:.1f}%")
                        
                        # Similarity visualization
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = similarity_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Similarity Score"},
                            delta = {'reference': 0.5},
                            gauge = {
                                'axis': {'range': [None, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "lightgray"},
                                    {'range': [0.3, 0.6], 'color': "yellow"},
                                    {'range': [0.6, 0.8], 'color': "orange"},
                                    {'range': [0.8, 1], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.9
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Text previews
                        st.write("**Text Comparison:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Text 1:**")
                            st.text_area("", value=text1, height=100, disabled=True)
                        with col2:
                            st.write("**Text 2:**")
                            st.text_area("", value=text2, height=100, disabled=True)
                    
                    except Exception as e:
                        st.error(f"Error calculating similarity: {e}")
            else:
                st.warning("Please enter both texts for comparison")
        
        st.markdown("---")
        
        # Batch embedding generation
        st.write("**Batch Embedding Generation**")
        batch_texts = st.text_area(
            "Enter multiple texts (one per line):",
            height=150,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        if st.button("🔢 Generate Batch Embeddings"):
            if batch_texts:
                texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Generating embeddings for {len(texts)} texts..."):
                        try:
                            embeddings = analyzer.generate_embeddings(texts)
                            
                            st.success(f"✅ Generated {len(embeddings)} embeddings!")
                            
                            # Create results DataFrame
                            results_data = []
                            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                                results_data.append({
                                    "Index": i + 1,
                                    "Text": text[:100] + "..." if len(text) > 100 else text,
                                    "Dimensions": len(embedding),
                                    "Mean": np.array(embedding).mean(),
                                    "Std": np.array(embedding).std()
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download batch embeddings
                            batch_data = {
                                "texts": texts,
                                "embeddings": embeddings,
                                "model": analyzer.embedding_model,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            st.download_button(
                                label="📥 Download Batch Embeddings JSON",
                                data=json.dumps(batch_data, indent=2),
                                file_name=f"batch_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        except Exception as e:
                            st.error(f"Error generating batch embeddings: {e}")
                else:
                    st.warning("No valid texts found")
            else:
                st.warning("Please enter texts for batch processing")
    
    with tab5:
        st.subheader("Analysis Results & Statistics")
        st.info("View aggregated analysis results and statistics")
        
        # Placeholder for future implementation
        st.write("This section will show:")
        st.write("- Historical analysis trends")
        st.write("- Issue type distributions")
        st.write("- Geographic analysis patterns")
        st.write("- Performance metrics")
        
        # Sample statistics
        st.subheader("Sample Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total MSOAs", len(mapper.get_all_msoa_codes()))
        with col2:
            st.metric("Local Authorities", len(mapper.get_local_authorities()))
        with col3:
            st.metric("Issue Categories", len(analyzer.issue_categories))

def community_survey_page():
    """Community Life Survey data analysis page"""
    st.title("📋 Community Life Survey Analysis")
    st.markdown("Analyze community engagement and social cohesion data from the Community Life Survey")
    
    # Get data connector
    connector = st.session_state.unified_data_connector
    
    # Check if Community Life Survey data is available
    survey_data = connector.get_community_survey_data()
    if survey_data is None or survey_data.empty:
        st.error("❌ Community Life Survey data is not available. Please ensure the data file is loaded.")
        return
    
    # Get summary
    summary = connector.get_community_survey_summary()
    
    # Display summary metrics
    st.subheader("📊 Survey Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Questions", summary.get('total_questions', 0))
    with col2:
        st.metric("Total Sheets", summary.get('total_sheets', 0))
    with col3:
        st.metric("Total Responses", summary.get('total_responses', 0))
    with col4:
        st.metric("Local Authorities", summary.get('unique_lads', 0))
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filter Options")
    
    # Question filter - show all questions for better user experience
    all_questions = connector.get_all_survey_questions()
    
    # Add search functionality for questions
    st.sidebar.write("**🔍 Search Questions:**")
    question_search = st.sidebar.text_input("Type to search questions:", placeholder="e.g., satisfaction, trust, community")
    
    # Filter questions based on search
    if question_search:
        filtered_questions = [q for q in all_questions if question_search.lower() in q.lower()]
        if filtered_questions:
            question_options = ["All Questions"] + filtered_questions
            st.sidebar.write(f"Found {len(filtered_questions)} matching questions")
        else:
            question_options = ["All Questions"] + all_questions
            st.sidebar.warning("No questions found matching your search. Showing all questions.")
    else:
        question_options = ["All Questions"] + all_questions
    
    selected_question = st.sidebar.selectbox("Select Survey Question", question_options, help="Choose a specific survey question to analyze, or select 'All Questions' to see data from all questions", key="community_survey_question_selectbox")
    
    # LAD filter
    lad_column = survey_data.columns[1]  # Column B should be LAD names
    unique_lads = sorted(survey_data[lad_column].dropna().unique())
    lad_options = ["All Local Authorities"] + unique_lads
    selected_lad = st.sidebar.selectbox("Select Local Authority District", lad_options, help="Choose a specific Local Authority District to analyze, or select 'All Local Authorities' to see data from all areas", key="community_survey_lad_selectbox")
    
    # Filter data based on selections
    filtered_data = survey_data.copy()
    
    if selected_question != "All Questions":
        filtered_data = filtered_data[filtered_data['question'] == selected_question]
    
    if selected_lad != "All Local Authorities":
        filtered_data = filtered_data[filtered_data[lad_column] == selected_lad]
    
    # Display filtered results
    st.subheader("📈 Analysis Results")
    
    if filtered_data.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Show data table
    st.subheader("📋 Survey Data")
    
    # Prepare display data
    display_data = filtered_data.copy()
    
    # Show only relevant columns for display
    display_columns = [lad_column, 'question'] + [col for col in display_data.columns if col not in [lad_column, 'question', 'sheet_name']]
    display_data = display_data[display_columns]
    
    st.dataframe(display_data, use_container_width=True)
    
    # Question analysis
    if selected_question == "All Questions":
        st.subheader("📊 Question Analysis")
        
        # Top questions chart
        question_counts = filtered_data['question'].value_counts().head(10)
        
        fig = px.bar(
            x=question_counts.values,
            y=question_counts.index,
            orientation='h',
            title="Top 10 Most Common Questions",
            labels={'x': 'Number of Responses', 'y': 'Question'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # LAD analysis
    if selected_lad == "All Local Authorities":
        st.subheader("🏛️ Local Authority Analysis")
        
        # Top LADs by response count
        lad_counts = filtered_data[lad_column].value_counts().head(10)
        
        fig = px.bar(
            x=lad_counts.values,
            y=lad_counts.index,
            orientation='h',
            title="Top 10 Local Authorities by Response Count",
            labels={'x': 'Number of Responses', 'y': 'Local Authority'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis for specific selections
    if selected_question != "All Questions" and selected_lad != "All Local Authorities":
        st.subheader("🔍 Detailed Analysis")
        
        # Get specific data
        specific_data = connector.get_lad_survey_data(selected_lad)
        question_data = specific_data[specific_data['question'] == selected_question]
        
        if not question_data.empty:
            st.write(f"**Question:** {selected_question}")
            st.write(f"**Local Authority:** {selected_lad}")
            
            # Show response data
            response_columns = [col for col in question_data.columns if col not in [lad_column, 'question', 'sheet_name']]
            if response_columns:
                st.write("**Response Data:**")
                response_data = question_data[response_columns].iloc[0]
                for col, value in response_data.items():
                    if pd.notna(value):
                        st.write(f"- **{col}:** {value}")
    
    # Export options
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Filtered Data (CSV)"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"community_survey_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Summary Report"):
            # Create summary report
            report_data = {
                'total_questions': summary.get('total_questions', 0),
                'total_responses': summary.get('total_responses', 0),
                'unique_lads': summary.get('unique_lads', 0),
                'selected_filters': {
                    'question': selected_question,
                    'local_authority': selected_lad
                },
                'filtered_results': len(filtered_data)
            }
            
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="Download Report",
                data=report_json,
                file_name=f"community_survey_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
