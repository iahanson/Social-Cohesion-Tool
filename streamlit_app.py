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

# Try to import streamlit_folium, fallback if not available
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("streamlit-folium not installed. Maps will be displayed as static images. Install with: pip install streamlit-folium")

# Import our custom modules
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator
from src.alert_system import AlertSystem
from src.good_neighbours_connector import GoodNeighboursConnector
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

# Initialize session state
if 'early_warning_system' not in st.session_state:
    st.session_state.early_warning_system = EarlyWarningSystem()
if 'sentiment_mapping' not in st.session_state:
    st.session_state.sentiment_mapping = SentimentMapping()
if 'intervention_tool' not in st.session_state:
    st.session_state.intervention_tool = InterventionTool()
if 'engagement_simulator' not in st.session_state:
    st.session_state.engagement_simulator = EngagementSimulator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'good_neighbours_connector' not in st.session_state:
    st.session_state.good_neighbours_connector = GoodNeighboursConnector()

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏘️ Social Cohesion Monitoring System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a component:",
        [
            "📊 Dashboard Overview",
            "🚨 Early Warning System",
            "🗺️ Sentiment & Trust Mapping",
            "🤝 Good Neighbours Trust Data",
            "💡 Intervention Recommendations",
            "🎯 Engagement Simulator",
            "📧 Alert Management",
            "🤖 GenAI Text Analysis",
            "⚙️ System Settings"
        ]
    )
    
    # Route to appropriate page
    if page == "📊 Dashboard Overview":
        dashboard_overview()
    elif page == "🚨 Early Warning System":
        early_warning_page()
    elif page == "🗺️ Sentiment & Trust Mapping":
        sentiment_mapping_page()
    elif page == "🤝 Good Neighbours Trust Data":
        good_neighbours_page()
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
        gn_data = st.session_state.good_neighbours_connector.load_social_trust_data()
        gn_summary = st.session_state.good_neighbours_connector.get_social_trust_summary()
    
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
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Level Distribution")
        risk_data = ew_data['data']['risk_level'].value_counts()
        fig = px.pie(
            values=risk_data.values,
            names=risk_data.index,
            color_discrete_map={
                'Low': '#4caf50',
                'Medium': '#ff9800',
                'High': '#ff5722',
                'Critical': '#f44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Trust vs Deprivation")
        fig = px.scatter(
            sm_data['data'],
            x='deprivation_composite',
            y='social_trust_composite',
            color='local_authority',
            size='population',
            hover_data=['msoa_name', 'msoa_code'],
            title="Social Trust vs Deprivation by Local Authority"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Assessment Summary
    st.subheader("🚨 Risk Assessment Summary")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Risk Assessment", help="Re-run risk assessment with latest data"):
            st.rerun()
    with col2:
        if st.button("📊 View Detailed Analysis", help="Go to detailed risk assessment page"):
            st.session_state.page = "🚨 Early Warning System"
            st.rerun()
    with col3:
        if st.button("📈 Export Risk Data", help="Download risk assessment data"):
            # Create downloadable CSV
            csv_data = ew_data['data'][['msoa_code', 'local_authority', 'risk_score', 'risk_level', 
                                       'unemployment_rate', 'crime_rate', 'social_trust_score', 
                                       'community_cohesion']].to_csv(index=False)
            st.download_button(
                label="Download Risk Assessment Data",
                data=csv_data,
                file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Risk score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            ew_data['data'],
            x='risk_score',
            nbins=20,
            title="Risk Score Distribution Across All Areas",
            labels={'risk_score': 'Risk Score', 'count': 'Number of Areas'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Risk Areas")
        top_risk_areas = ew_data['data'].nlargest(5, 'risk_score')[
            ['msoa_code', 'local_authority', 'risk_score', 'risk_level']
        ]
        
        # Display as a styled table
        for idx, row in top_risk_areas.iterrows():
            risk_color = {
                'Low': '#4caf50',
                'Medium': '#ff9800', 
                'High': '#ff5722',
                'Critical': '#f44336'
            }.get(row['risk_level'], '#666666')
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {risk_color}; margin-bottom: 0.5rem;">
                <strong>{row['msoa_code']}</strong> - {row['local_authority']}<br>
                Risk Score: {row['risk_score']:.3f} | Level: {row['risk_level']}
            </div>
            """, unsafe_allow_html=True)
    
    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    if ew_data['alerts']:
        for alert in ew_data['alerts'][:5]:  # Show top 5 alerts
            priority_class = f"alert-{alert['priority'].lower()}"
            st.markdown(f"""
            <div class="metric-card {priority_class}">
                <strong>{alert['type']}</strong> - {alert['priority']} Priority<br>
                MSOA: {alert['msoa_code']} | LA: {alert['local_authority']}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent alerts")

def early_warning_page():
    """Early warning system page"""
    st.header("🚨 Early Warning System")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Full Analysis", "Risk Assessment", "Anomaly Detection", "Area Profile"]
        )
    
    with col2:
        if analysis_type == "Area Profile":
            msoa_code = st.text_input("MSOA Code:", value="E02000001")
        else:
            n_areas = st.slider("Number of Areas:", 10, 200, 100)
    
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
            ["Trust Map", "Cohesion Map", "Sentiment Map", "Correlation Analysis", "Cohesion Dashboard"]
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
                            size='population',
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
                        size='population',
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
                        size='population',
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
                        size='population',
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
                        size='population',
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
                        size='population',
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
                        size='population',
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
            ["Overview", "MSOA Lookup", "Top Trust Areas", "Lowest Trust Areas", "Trust Distribution"]
        )
    
    with col2:
        if analysis_type == "MSOA Lookup":
            msoa_code = st.text_input("MSOA Code:", value="E02000001")
        elif analysis_type in ["Top Trust Areas", "Lowest Trust Areas"]:
            n_areas = st.slider("Number of Areas:", 5, 50, 10)
    
    # Run analysis
    if st.button("Run Analysis"):
        with st.spinner("Loading Good Neighbours data..."):
            connector = st.session_state.good_neighbours_connector
            
            if analysis_type == "Overview":
                # Load data and show summary
                df = connector.load_social_trust_data()
                summary = connector.get_social_trust_summary()
                
                if df is not None and summary is not None:
                    st.subheader("📊 Data Overview")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total MSOAs", summary['total_msoas'])
                    
                    with col2:
                        st.metric("Average Net Trust", f"{summary['average_net_trust']:.2f}")
                    
                    with col3:
                        st.metric("Highest Trust", f"{summary['net_trust_range']['max']:.2f}")
                    
                    with col4:
                        st.metric("Lowest Trust", f"{summary['net_trust_range']['min']:.2f}")
                    
                    # Trust distribution
                    st.subheader("Trust Distribution")
                    trust_dist = summary['net_trust_distribution']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Trust", trust_dist['positive_trust'])
                    with col2:
                        st.metric("Negative Trust", trust_dist['negative_trust'])
                    with col3:
                        st.metric("Neutral Trust", trust_dist['neutral_trust'])
                    
                    # Net trust distribution chart
                    st.subheader("Net Trust Distribution")
                    fig = px.histogram(
                        df,
                        x='Net_trust',
                        nbins=30,
                        title="Distribution of Net Trust Scores",
                        labels={'Net_trust': 'Net Trust Score', 'count': 'Number of MSOAs'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trust components
                    st.subheader("Trust Components")
                    components = summary['trust_components']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Always/Usually Trust", f"{components['average_always_usually_trust']:.1f}%")
                    with col2:
                        st.metric("Usually/Almost Always Careful", f"{components['average_usually_almost_always_careful']:.1f}%")
                    
                    # Scatter plot of trust components
                    fig = px.scatter(
                        df,
                        x='always_trust OR usually_trust',
                        y='usually_careful OR almost_always_careful',
                        color='Net_trust',
                        hover_data=['MSOA_name', 'MSOA_code'],
                        title="Trust Components Relationship",
                        color_continuous_scale='RdYlGn',
                        labels={
                            'always_trust OR usually_trust': 'Always/Usually Trust (%)',
                            'usually_careful OR almost_always_careful': 'Usually/Almost Always Careful (%)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to load Good Neighbours data")
            
            elif analysis_type == "MSOA Lookup":
                trust_data = connector.get_social_trust_for_msoa(msoa_code)
                
                if trust_data:
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
                top_areas = connector.get_top_trust_msoas(n_areas)
                
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
                lowest_areas = connector.get_lowest_trust_msoas(n_areas)
                
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
                df = connector.load_social_trust_data()
                summary = connector.get_social_trust_summary()
                
                if df is not None and summary is not None:
                    st.subheader("Trust Score Distribution Analysis")
                    
                    # Decile analysis
                    df['trust_decile'] = pd.qcut(df['Net_trust'], q=10, labels=False, duplicates='drop') + 1
                    
                    decile_stats = df.groupby('trust_decile')['Net_trust'].agg(['count', 'mean', 'min', 'max']).reset_index()
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
                        x='always_trust OR usually_trust',
                        y='usually_careful OR almost_always_careful',
                        color='Net_trust',
                        hover_data=['MSOA_name', 'MSOA_code'],
                        title="Trust vs Caution Relationship",
                        color_continuous_scale='RdYlGn',
                        labels={
                            'always_trust OR usually_trust': 'Always/Usually Trust (%)',
                            'usually_careful OR almost_always_careful': 'Usually/Almost Always Careful (%)'
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
        ["Custom Scenario", "Predefined Scenarios", "Optimization"]
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
            ["overall", "trust", "cohesion", "sentiment"]
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
                ["survey", "social_media", "report", "interview", "feedback", "other"]
            )
        with col2:
            analysis_type = st.selectbox(
                "Analysis focus:",
                ["comprehensive", "social_cohesion_only", "location_focused", "sentiment_only"]
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

if __name__ == "__main__":
    main()
