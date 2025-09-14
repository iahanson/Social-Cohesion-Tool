#!/usr/bin/env python3
"""
Social Cohesion Monitoring System
A comprehensive tool for identifying areas of low social cohesion and rising tensions,
and prioritizing interventions for local stakeholders.
"""

import click
import json
import pandas as pd
from src.data_aggregator import MSOADataAggregator
from src.msoa_search import MSOASearch
from src.imd_connector import IMDConnector
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator
from src.alert_system import AlertSystem
from src.good_neighbours_connector import GoodNeighboursConnector
from src.data_config import get_data_config

@click.group()
def cli():
    """Social Cohesion Monitoring System - Early warning and intervention tools"""
    pass

# Original MSOA lookup commands
@cli.group()
def msoa():
    """MSOA data lookup and analysis"""
    pass

@msoa.command()
@click.option('--postcode', '-p', help='UK postcode to look up')
@click.option('--msoa-code', '-m', help='MSOA code to look up directly')
@click.option('--output', '-o', default='console', help='Output format: console, json, csv')
def lookup(postcode, msoa_code, output):
    """Look up data for a specific MSOA by postcode or MSOA code"""
    if not postcode and not msoa_code:
        click.echo("Please provide either a postcode (-p) or MSOA code (-m)")
        return
    
    aggregator = MSOADataAggregator()
    
    if postcode:
        msoa_info = aggregator.get_msoa_by_postcode(postcode)
    else:
        msoa_info = aggregator.get_msoa_by_code(msoa_code)
    
    if msoa_info:
        aggregator.display_results(msoa_info, output)
    else:
        click.echo("MSOA not found")

@msoa.command()
def sources():
    """List available data sources"""
    click.echo("Available data sources:")
    click.echo("1. Indices of Multiple Deprivation (IMD) - England 2019")
    click.echo("2. ONS Postcode Directory")
    click.echo("3. Composite UK IMD Dataset (mySociety)")

# Early Warning System commands
@cli.group()
def warning():
    """Early warning system for social tension detection"""
    pass

@warning.command()
@click.option('--areas', '-a', default=100, help='Number of areas to analyze')
@click.option('--output', '-o', default='console', help='Output format: console, json')
def analyze(areas, output):
    """Run early warning analysis on areas"""
    click.echo("Running early warning analysis...")
    
    ew_system = EarlyWarningSystem()
    results = ew_system.run_full_analysis()
    
    if output == 'json':
        click.echo(json.dumps(results['summary'], indent=2))
    else:
        click.echo(f"Analysis complete!")
        click.echo(f"Total areas analyzed: {results['summary']['total_areas']}")
        click.echo(f"Critical risk areas: {results['summary']['critical_risk_areas']}")
        click.echo(f"High risk areas: {results['summary']['high_risk_areas']}")
        click.echo(f"Anomalous areas: {results['summary']['anomalous_areas']}")
        click.echo(f"Total alerts generated: {results['summary']['total_alerts']}")
        
        if results['alerts']:
            click.echo("\nGenerated Alerts:")
            for alert in results['alerts'][:5]:  # Show top 5
                click.echo(f"- {alert['type']}: {alert['message']}")

@warning.command()
@click.option('--msoa-code', '-m', required=True, help='MSOA code to analyze')
def profile(msoa_code):
    """Get detailed risk profile for a specific MSOA"""
    click.echo(f"Analyzing risk profile for MSOA: {msoa_code}")
    
    ew_system = EarlyWarningSystem()
    data = ew_system.generate_sample_data()
    profile = ew_system.get_risk_factors(data, msoa_code)
    
    if 'error' in profile:
        click.echo(f"Error: {profile['error']}")
        return
    
    click.echo(f"\nRisk Profile for {msoa_code}:")
    click.echo(f"Local Authority: {profile['local_authority']}")
    click.echo(f"Overall Risk Score: {profile['overall_risk_score']:.2f}")
    click.echo(f"Risk Level: {profile['risk_level']}")
    
    click.echo("\nTop Risk Factors:")
    for factor, score in profile['top_risk_factors']:
        click.echo(f"- {factor}: {score:.2f}")
    
    click.echo("\nRecommendations:")
    for rec in profile['recommendations']:
        click.echo(f"- {rec}")

# Sentiment Mapping commands
@cli.group()
def sentiment():
    """Sentiment and social trust mapping"""
    pass

@sentiment.command()
@click.option('--areas', '-a', default=50, help='Number of areas to analyze')
@click.option('--output', '-o', default='console', help='Output format: console, json')
def map(areas, output):
    """Generate sentiment and trust mapping analysis"""
    click.echo("Generating sentiment mapping analysis...")
    
    sm_system = SentimentMapping()
    results = sm_system.run_full_analysis()
    
    if output == 'json':
        click.echo(json.dumps(results['summary'], indent=2))
    else:
        click.echo(f"Sentiment mapping complete!")
        click.echo(f"Total areas analyzed: {results['summary']['total_areas']}")
        click.echo(f"Average trust score: {results['summary']['average_trust_score']:.1f}/10")
        click.echo(f"Average cohesion score: {results['summary']['average_cohesion_score']:.1f}/10")
        click.echo(f"Average sentiment score: {results['summary']['average_sentiment_score']:.1f}/10")

@sentiment.command()
@click.option('--msoa-code', '-m', required=True, help='MSOA code to analyze')
def profile(msoa_code):
    """Get detailed sentiment profile for a specific MSOA"""
    click.echo(f"Analyzing sentiment profile for MSOA: {msoa_code}")
    
    sm_system = SentimentMapping()
    data = sm_system.generate_sample_data()
    profile = sm_system.analyze_area_profile(data, msoa_code)
    
    if 'error' in profile:
        click.echo(f"Error: {profile['error']}")
        return
    
    click.echo(f"\nSentiment Profile for {msoa_code}:")
    click.echo(f"MSOA Name: {profile['msoa_name']}")
    click.echo(f"Local Authority: {profile['local_authority']}")
    click.echo(f"Population: {profile['population']:,}")
    
    click.echo(f"\nSocial Trust Score: {profile['scores']['social_trust']['score']}/10")
    click.echo(f"Community Cohesion Score: {profile['scores']['community_cohesion']['score']}/10")
    click.echo(f"Sentiment Score: {profile['scores']['sentiment']['score']}/10")
    click.echo(f"Overall Cohesion Score: {profile['scores']['overall_cohesion']}/10")
    
    if profile['challenges']:
        click.echo("\nChallenges:")
        for challenge in profile['challenges']:
            click.echo(f"- {challenge}")
    
    if profile['strengths']:
        click.echo("\nStrengths:")
        for strength in profile['strengths']:
            click.echo(f"- {strength}")
    
    if profile['recommendations']:
        click.echo("\nRecommendations:")
        for rec in profile['recommendations']:
            click.echo(f"- {rec}")

# Good Neighbours Social Trust commands
@cli.group()
def trust():
    """Good Neighbours social trust data analysis"""
    pass

@trust.command()
@click.option('--output', '-o', default='console', help='Output format: console, json')
def summary(output):
    """Get summary of social trust data"""
    click.echo("Loading Good Neighbours social trust data...")
    
    connector = GoodNeighboursConnector()
    summary = connector.get_social_trust_summary()
    
    if summary is None:
        click.echo("Error: Failed to load social trust data")
        return
    
    if output == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"Social Trust Data Summary:")
        click.echo(f"Total MSOAs: {summary['total_msoas']}")
        click.echo(f"Average Net Trust: {summary['average_net_trust']:.2f}")
        click.echo(f"Trust Range: {summary['net_trust_range']['min']:.2f} to {summary['net_trust_range']['max']:.2f}")
        click.echo(f"Standard Deviation: {summary['net_trust_range']['std']:.2f}")
        
        click.echo(f"\nTrust Distribution:")
        dist = summary['net_trust_distribution']
        click.echo(f"Positive Trust: {dist['positive_trust']} MSOAs")
        click.echo(f"Negative Trust: {dist['negative_trust']} MSOAs")
        click.echo(f"Neutral Trust: {dist['neutral_trust']} MSOAs")
        
        click.echo(f"\nHighest Trust MSOA:")
        highest = summary['highest_trust_msoa']
        click.echo(f"  {highest['name']} ({highest['code']}) - Net Trust: {highest['net_trust']:.2f}")
        
        click.echo(f"\nLowest Trust MSOA:")
        lowest = summary['lowest_trust_msoa']
        click.echo(f"  {lowest['name']} ({lowest['code']}) - Net Trust: {lowest['net_trust']:.2f}")

@trust.command()
@click.option('--msoa-code', '-m', required=True, help='MSOA code to look up')
def lookup(msoa_code):
    """Get social trust data for a specific MSOA"""
    click.echo(f"Looking up social trust data for MSOA: {msoa_code}")
    
    connector = GoodNeighboursConnector()
    trust_data = connector.get_social_trust_for_msoa(msoa_code)
    
    if trust_data is None:
        click.echo(f"Error: No social trust data found for MSOA {msoa_code}")
        return
    
    click.echo(f"\nSocial Trust Data for {msoa_code}:")
    click.echo(f"MSOA Name: {trust_data['msoa_name']}")
    click.echo(f"Net Trust Score: {trust_data['net_trust']:.2f}")
    click.echo(f"Always/Usually Trust: {trust_data['always_usually_trust']:.1f}%")
    click.echo(f"Usually/Almost Always Careful: {trust_data['usually_almost_always_careful']:.1f}%")
    
    # Interpretation
    if trust_data['net_trust'] > 0:
        click.echo(f"\nInterpretation: Positive net trust score indicates higher trust than caution")
    elif trust_data['net_trust'] < 0:
        click.echo(f"\nInterpretation: Negative net trust score indicates higher caution than trust")
    else:
        click.echo(f"\nInterpretation: Neutral net trust score indicates balanced trust and caution")

@trust.command()
@click.option('--top', '-t', default=10, help='Number of top trust areas to show')
def top(top):
    """Show top trust areas"""
    click.echo(f"Loading top {top} trust areas...")
    
    connector = GoodNeighboursConnector()
    top_areas = connector.get_top_trust_msoas(top)
    
    if not top_areas:
        click.echo("Error: Failed to load top trust areas")
        return
    
    click.echo(f"\nTop {top} Trust Areas:")
    click.echo("-" * 80)
    click.echo(f"{'Rank':<4} {'MSOA Name':<30} {'MSOA Code':<12} {'Net Trust':<10} {'Trust %':<8} {'Careful %':<10}")
    click.echo("-" * 80)
    
    for i, area in enumerate(top_areas, 1):
        click.echo(f"{i:<4} {area['msoa_name'][:29]:<30} {area['msoa_code']:<12} {area['net_trust']:<10.2f} {area['always_usually_trust']:<8.1f} {area['usually_almost_always_careful']:<10.1f}")

@trust.command()
@click.option('--bottom', '-b', default=10, help='Number of lowest trust areas to show')
def lowest(bottom):
    """Show lowest trust areas"""
    click.echo(f"Loading lowest {bottom} trust areas...")
    
    connector = GoodNeighboursConnector()
    lowest_areas = connector.get_lowest_trust_msoas(bottom)
    
    if not lowest_areas:
        click.echo("Error: Failed to load lowest trust areas")
        return
    
    click.echo(f"\nLowest {bottom} Trust Areas:")
    click.echo("-" * 80)
    click.echo(f"{'Rank':<4} {'MSOA Name':<30} {'MSOA Code':<12} {'Net Trust':<10} {'Trust %':<8} {'Careful %':<10}")
    click.echo("-" * 80)
    
    for i, area in enumerate(lowest_areas, 1):
        click.echo(f"{i:<4} {area['msoa_name'][:29]:<30} {area['msoa_code']:<12} {area['net_trust']:<10.2f} {area['always_usually_trust']:<8.1f} {area['usually_almost_always_careful']:<10.1f}")

# Intervention Tool commands
@cli.group()
def intervention():
    """Intervention effectiveness and recommendations"""
    pass

@intervention.command()
@click.option('--msoa-code', '-m', help='MSOA code for recommendations')
@click.option('--recommendations', '-r', default=5, help='Number of recommendations')
def recommend(msoa_code, recommendations):
    """Get intervention recommendations for an area"""
    click.echo("Analyzing intervention options...")
    
    int_tool = InterventionTool()
    
    # Create sample target area
    target_area = {
        'population': 8500,
        'unemployment_rate': 8.5,
        'crime_rate': 75,
        'income_deprivation': 25,
        'education_attainment': 68,
        'ethnic_diversity': 35,
        'age_dependency_ratio': 0.65,
        'housing_stress': 40,
        'social_trust_pre': 5.8,
        'community_cohesion_pre': 6.1,
        'volunteer_rate_pre': 15
    }
    
    results = int_tool.run_full_analysis(target_area)
    
    click.echo(f"\nIntervention Recommendations:")
    click.echo(f"Based on {results['summary']['total_cases_in_database']} historical cases")
    
    for i, rec in enumerate(results['recommendations'][:recommendations], 1):
        click.echo(f"\n{i}. {rec['intervention_type']}")
        click.echo(f"   Expected Success Score: {rec['expected_success_score']:.2f}")
        click.echo(f"   Trust Improvement: {rec['expected_trust_improvement']:.2f}")
        click.echo(f"   Cohesion Improvement: {rec['expected_cohesion_improvement']:.2f}")
        click.echo(f"   Average Funding: £{rec['average_funding_required']:,.0f}")
        click.echo(f"   Duration: {rec['average_duration_months']:.0f} months")
        click.echo(f"   Cost Effectiveness: {rec['cost_effectiveness']:.3f}")
        click.echo(f"   Evidence Base: {rec['evidence_base']}")

@intervention.command()
@click.option('--intervention-type', '-t', required=True, help='Intervention type to analyze')
def analyze(intervention_type):
    """Analyze effectiveness of a specific intervention type"""
    click.echo(f"Analyzing effectiveness of: {intervention_type}")
    
    int_tool = InterventionTool()
    analysis = int_tool.analyze_intervention_effectiveness(intervention_type)
    
    if 'error' in analysis:
        click.echo(f"Error: {analysis['error']}")
        return
    
    click.echo(f"\nIntervention Analysis: {intervention_type}")
    click.echo(f"Total cases: {analysis['total_cases']}")
    click.echo(f"Average success score: {analysis['average_success_score']:.2f}")
    click.echo(f"Success rate: {analysis['success_rate']:.1%}")
    click.echo(f"Average funding: £{analysis['average_funding']:,.0f}")
    click.echo(f"Average duration: {analysis['average_duration']:.0f} months")
    click.echo(f"Cost effectiveness: {analysis['cost_effectiveness']:.3f}")
    
    click.echo(f"\nOutcomes:")
    click.echo(f"Trust improvement: {analysis['outcomes']['trust_improvement']['mean']:.2f} ± {analysis['outcomes']['trust_improvement']['std']:.2f}")
    click.echo(f"Cohesion improvement: {analysis['outcomes']['cohesion_improvement']['mean']:.2f} ± {analysis['outcomes']['cohesion_improvement']['std']:.2f}")
    click.echo(f"Volunteer improvement: {analysis['outcomes']['volunteer_improvement']['mean']:.2f} ± {analysis['outcomes']['volunteer_improvement']['std']:.2f}")

# Engagement Simulator commands
@cli.group()
def simulator():
    """Community engagement impact simulator"""
    pass

@simulator.command()
@click.option('--budget', '-b', default=1000, help='Available budget')
@click.option('--target', '-t', default='overall', help='Target outcome: overall, trust, cohesion, sentiment')
def optimize(budget, target):
    """Optimize intervention mix for given budget and target"""
    click.echo(f"Optimizing interventions for budget: £{budget}, target: {target}")
    
    simulator = EngagementSimulator()
    
    # Sample baseline area
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
    
    result = simulator.optimize_intervention_mix(baseline_area, budget, target)
    
    if result['optimization_successful']:
        click.echo("Optimization completed successfully!")
        
        simulation = result['simulation_result']
        click.echo(f"\nExpected Improvements:")
        click.echo(f"Trust: {simulation['improvements']['trust_improvement']:.2f}")
        click.echo(f"Cohesion: {simulation['improvements']['cohesion_improvement']:.2f}")
        click.echo(f"Sentiment: {simulation['improvements']['sentiment_improvement']:.2f}")
        click.echo(f"Overall: {simulation['improvements']['overall_improvement']:.2f}")
        
        click.echo(f"\nOptimized Investment Distribution:")
        interventions = result['optimized_interventions']
        for intervention, investment in interventions.items():
            if investment > 0:
                click.echo(f"- {intervention}: £{investment:.0f}")
    else:
        click.echo(f"Optimization failed: {result['error']}")

# Alert System commands
@cli.group()
def alerts():
    """Alert system management"""
    pass

@alerts.command()
def test():
    """Test alert system configuration"""
    click.echo("Testing alert system...")
    
    alert_system = AlertSystem()
    test_result = alert_system.test_alert_system()
    
    click.echo(f"Test alert created: {test_result['test_alert_created']}")
    click.echo(f"Email test: {'✅' if test_result['email_test'] else '❌'}")
    click.echo(f"SMS test: {'✅' if test_result['sms_test'] else '❌'}")
    
    if test_result['email_error']:
        click.echo(f"Email error: {test_result['email_error']}")
    if test_result['sms_error']:
        click.echo(f"SMS error: {test_result['sms_error']}")

@alerts.command()
@click.option('--hours', '-h', default=24, help='Time window for alert summary')
def summary(hours):
    """Get alert summary for specified time window"""
    click.echo(f"Alert summary for last {hours} hours:")
    
    alert_system = AlertSystem()
    summary = alert_system.create_alert_summary_report(hours)
    
    if summary['total_alerts'] == 0:
        click.echo("No alerts in the specified time window")
        return
    
    click.echo(f"Total alerts: {summary['total_alerts']}")
    click.echo(f"Success rate: {summary['success_rate']:.1%}")
    click.echo(f"Email alerts: {summary['email_alerts']}")
    click.echo(f"SMS alerts: {summary['sms_alerts']}")
    
    if summary['priorities']:
        click.echo("\nAlerts by priority:")
        for priority, count in summary['priorities'].items():
            click.echo(f"- {priority}: {count}")

# Main system commands
@cli.command()
def dashboard():
    """Launch Streamlit dashboard"""
    import subprocess
    import sys
    
    click.echo("Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

@cli.group()
def data():
    """Data source configuration and management"""
    pass

@data.command()
def status():
    """Show data source configuration status"""
    click.echo("Data Source Configuration Status")
    click.echo("=" * 40)
    
    config = get_data_config()
    status_summary = config.get_status_summary()
    
    click.echo(f"Total Sources: {status_summary['total_sources']}")
    click.echo(f"Enabled Sources: {status_summary['enabled_sources']}")
    click.echo(f"Real Data Sources: {status_summary['real_data_sources']}")
    click.echo(f"Dummy Data Sources: {status_summary['dummy_data_sources']}")
    
    click.echo("\nSource Details:")
    click.echo("-" * 20)
    
    for source_name, source_info in status_summary['sources'].items():
        status_icon = "✅" if source_info['enabled'] else "❌"
        data_type = "REAL" if source_info['use_real_data'] else "DUMMY"
        available = "✓" if source_info['real_data_available'] else "✗"
        
        click.echo(f"{status_icon} {source_name.upper()}: {data_type} ({available})")
        if source_info['data_url']:
            click.echo(f"    URL: {source_info['data_url']}")
        if source_info['local_file_path']:
            click.echo(f"    File: {source_info['local_file_path']}")

@data.command()
@click.option('--source', required=True, help='Data source name (e.g., imd_data, community_life_survey)')
@click.option('--use-real-data', is_flag=True, help='Enable real data for this source')
@click.option('--use-dummy-data', is_flag=True, help='Enable dummy data for this source')
def configure(source, use_real_data, use_dummy_data):
    """Configure data source settings"""
    if use_real_data and use_dummy_data:
        click.echo("Error: Cannot enable both real and dummy data simultaneously")
        return
    
    if not use_real_data and not use_dummy_data:
        click.echo("Error: Must specify either --use-real-data or --use-dummy-data")
        return
    
    config = get_data_config()
    source_config = config.get_config(source)
    
    if use_real_data:
        source_config.use_real_data = True
        click.echo(f"✅ Configured {source} to use REAL data")
    else:
        source_config.use_real_data = False
        click.echo(f"✅ Configured {source} to use DUMMY data")
    
    click.echo(f"Note: To make this permanent, set {source.upper()}_USE_REAL_DATA={'true' if use_real_data else 'false'} in your .env file")

@cli.command()
def status():
    """Show system status and configuration"""
    click.echo("Social Cohesion Monitoring System Status")
    click.echo("=" * 50)
    
    # Check data configuration
    try:
        config = get_data_config()
        status_summary = config.get_status_summary()
        click.echo(f"Data Configuration: {status_summary['real_data_sources']} real, {status_summary['dummy_data_sources']} dummy")
    except Exception as e:
        click.echo(f"Data Configuration: Error - {e}")
    
    # Check components
    components = {
        'Early Warning System': EarlyWarningSystem(),
        'Sentiment Mapping': SentimentMapping(),
        'Intervention Tool': InterventionTool(),
        'Engagement Simulator': EngagementSimulator(),
        'Alert System': AlertSystem()
    }
    
    for name, component in components.items():
        click.echo(f"{name}: ✅ Initialized")
    
    # Check alert system configuration
    alert_system = AlertSystem()
    click.echo(f"\nAlert System Configuration:")
    click.echo(f"Email configured: {'✅' if alert_system.email_username else '❌'}")
    click.echo(f"SMS configured: {'✅' if alert_system.twilio_account_sid else '❌'}")
    click.echo(f"Critical risk threshold: {alert_system.thresholds['critical_risk']}")
    
    click.echo(f"\nUse 'python main.py data status' for detailed data source information")

if __name__ == '__main__':
    cli()
