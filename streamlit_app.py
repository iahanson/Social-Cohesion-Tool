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
            "💡 Intervention Recommendations",
            "🎯 Engagement Simulator",
            "📧 Alert Management",
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
    elif page == "💡 Intervention Recommendations":
        intervention_page()
    elif page == "🎯 Engagement Simulator":
        engagement_simulator_page()
    elif page == "📧 Alert Management":
        alert_management_page()
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Areas Monitored",
            value=ew_data['summary']['total_areas'],
            delta=f"{ew_data['summary']['critical_risk_areas']} critical"
        )
    
    with col2:
        st.metric(
            label="Average Trust Score",
            value=f"{sm_data['summary']['average_trust_score']:.1f}/10",
            delta=f"{sm_data['summary']['trust_range']['min']:.1f}-{sm_data['summary']['trust_range']['max']:.1f}"
        )
    
    with col3:
        st.metric(
            label="Intervention Cases",
            value=int_data['summary']['total_cases_in_database'],
            delta=f"{int_data['summary']['intervention_types_available']} types"
        )
    
    with col4:
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

def sentiment_mapping_page():
    """Sentiment mapping page"""
    st.header("🗺️ Sentiment & Trust Mapping")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        map_type = st.selectbox(
            "Map Type:",
            ["Trust Map", "Cohesion Map", "Sentiment Map", "Correlation Analysis"]
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
            
            elif map_type == "Correlation Analysis":
                st.subheader("Correlation Analysis")
                
                # Correlation heatmap
                corr_fig = results['visualizations']['correlation_heatmap']
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Scatter plot
                scatter_fig = results['visualizations']['deprivation_scatter']
                st.plotly_chart(scatter_fig, use_container_width=True)

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

if __name__ == "__main__":
    main()
