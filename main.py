#!/usr/bin/env python3
"""
Social Cohesion Monitoring System
A comprehensive tool for identifying areas of low social cohesion and rising tensions,
and prioritizing interventions for local stakeholders.
"""

import click
import json
import pandas as pd
from dotenv import load_dotenv
from src.unified_data_connector import UnifiedDataConnector
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator
from src.alert_system import AlertSystem
from src.genai_text_analyzer import GenAITextAnalyzer
from src.locality_mapper import LocalityMapper
from src.data_config import get_data_config

# Load environment variables from .env file
load_dotenv(override=True)

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
    
    mapper = LocalityMapper()
    connector = UnifiedDataConnector()
    
    if postcode:
        click.echo(f"Looking up postcode: {postcode}")
        msoa_info = mapper.postcode_to_msoa(postcode)
        if msoa_info and msoa_info.get('msoa_code'):
            msoa_code = msoa_info['msoa_code']
        else:
            click.echo("Postcode not found")
            return
    else:
        msoa_code = msoa_code
    
    # Get comprehensive data
    results = connector.get_msoa_data(msoa_code)
    
    if output == 'json':
        click.echo(json.dumps({msoa_code: {source: result.data for source, result in results.items() if result.success}}, indent=2))
    elif output == 'csv':
        csv_data = connector.export_data([msoa_code], 'csv')
        click.echo(csv_data)
    else:
        # Console output
        click.echo(f"\n=== MSOA Data Report: {msoa_code} ===")
        for source, result in results.items():
            if result.success:
                click.echo(f"\n📊 {source.upper()} Data:")
                for key, value in result.data.items():
                    click.echo(f"   {key}: {value}")
            else:
                click.echo(f"\n❌ {source.upper()}: {result.error_message}")

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
    
    connector = UnifiedDataConnector()
    results = connector.get_msoa_data(msoa_code, ['good_neighbours'])
    
    trust_result = results.get('good_neighbours')
    if not trust_result or not trust_result.success:
        click.echo(f"Error: No social trust data found for MSOA {msoa_code}")
        return
    
    trust_data = trust_result.data
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
    
    connector = UnifiedDataConnector()
    top_areas = connector.get_top_performing_msoas('net_trust', top, 'good_neighbours')
    
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
    
    connector = UnifiedDataConnector()
    # Get lowest by reversing the order (ascending instead of descending)
    all_areas = connector.get_top_performing_msoas('net_trust', 1000, 'good_neighbours')
    lowest_areas = sorted(all_areas, key=lambda x: x['net_trust'])[:bottom]
    
    if not lowest_areas:
        click.echo("Error: Failed to load lowest trust areas")
        return
    
    click.echo(f"\nLowest {bottom} Trust Areas:")
    click.echo("-" * 80)
    click.echo(f"{'Rank':<4} {'MSOA Name':<30} {'MSOA Code':<12} {'Net Trust':<10} {'Trust %':<8} {'Careful %':<10}")
    click.echo("-" * 80)
    
    for i, area in enumerate(lowest_areas, 1):
        click.echo(f"{i:<4} {area['msoa_name'][:29]:<30} {area['msoa_code']:<12} {area['net_trust']:<10.2f} {area['always_usually_trust']:<8.1f} {area['usually_almost_always_careful']:<10.1f}")

# GenAI Text Analysis commands
@cli.group()
def genai():
    """GenAI-powered text analysis for social cohesion issues"""
    pass

@genai.command()
@click.option('--text', '-t', help='Text to analyze')
@click.option('--file', '-f', help='File containing text to analyze')
@click.option('--source', '-s', default='unknown', help='Source of the text (survey, social_media, report)')
@click.option('--output', '-o', default='console', help='Output format: console, json, csv, summary')
def analyze(text, file, source, output):
    """Analyze text for social cohesion issues using GenAI"""
    if not text and not file:
        click.echo("Error: Please provide either --text or --file")
        return
    
    if text and file:
        click.echo("Error: Please provide either --text or --file, not both")
        return
    
    # Get text content
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
            click.echo(f"Error reading file: {e}")
            return
    else:
        text_content = text
    
    click.echo("Analyzing text with GenAI...")
    
    try:
        analyzer = GenAITextAnalyzer()
        result = analyzer.analyze_text(text_content, source)
        
        if output == 'json':
            click.echo(json.dumps(analyzer._result_to_dict(result), indent=2))
        elif output == 'csv':
            click.echo(analyzer.export_results([result], 'csv'))
        elif output == 'summary':
            click.echo(analyzer.export_results([result], 'summary'))
        else:
            # Console output
            click.echo(f"\n=== GenAI Text Analysis Results ===")
            click.echo(f"Text ID: {result.text_id}")
            click.echo(f"Source: {result.source}")
            click.echo(f"Timestamp: {result.timestamp}")
            click.echo(f"Total Issues: {result.total_issues}")
            click.echo(f"Critical: {result.critical_issues}, High: {result.high_issues}, Medium: {result.medium_issues}, Low: {result.low_issues}")
            
            click.echo(f"\nSummary: {result.summary}")
            
            if result.localities_found:
                click.echo(f"\nLocalities Found:")
                for locality in result.localities_found:
                    click.echo(f"  - {locality['name']} ({locality['type']}) -> MSOA: {locality.get('msoa_code', 'Not mapped')}")
            
            if result.issues:
                click.echo(f"\nIssues Identified:")
                for i, issue in enumerate(result.issues, 1):
                    click.echo(f"\n{i}. {issue.issue_type.upper()} - {issue.severity} Severity")
                    click.echo(f"   Description: {issue.description}")
                    click.echo(f"   Confidence: {issue.confidence:.2f}")
                    if issue.location_mentioned:
                        click.echo(f"   Location: {issue.location_mentioned}")
                    if issue.msoa_code:
                        click.echo(f"   MSOA: {issue.msoa_code}")
                    if issue.local_authority:
                        click.echo(f"   Local Authority: {issue.local_authority}")
                    if issue.keywords:
                        click.echo(f"   Keywords: {', '.join(issue.keywords)}")
                    if issue.context:
                        click.echo(f"   Context: {issue.context}")
            
            if result.recommendations:
                click.echo(f"\nRecommendations:")
                for i, rec in enumerate(result.recommendations, 1):
                    click.echo(f"{i}. {rec}")
    
    except Exception as e:
        click.echo(f"Error during analysis: {e}")
        click.echo("Please check your Azure OpenAI configuration in the .env file")

@genai.command()
@click.option('--locality', '-l', required=True, help='Locality to map to MSOA')
def map_locality(locality):
    """Map a locality to MSOA code"""
    click.echo(f"Mapping locality: {locality}")
    
    try:
        mapper = LocalityMapper()
        result = mapper.map_locality(locality)
        
        if result:
            click.echo(f"\nLocality Mapping Result:")
            click.echo(f"Name: {result.name}")
            click.echo(f"Type: {result.type}")
            click.echo(f"MSOA Code: {result.msoa_code}")
            click.echo(f"Local Authority: {result.local_authority}")
            click.echo(f"Region: {result.region}")
            click.echo(f"Confidence: {result.confidence:.2f}")
        else:
            click.echo(f"No mapping found for: {locality}")
            click.echo("Try searching for similar localities:")
            
            # Try fuzzy search
            search_results = mapper.search_localities(locality)
            if search_results:
                click.echo(f"\nSimilar localities found:")
                for result in search_results[:5]:  # Show top 5
                    click.echo(f"  - {result.name} ({result.type}) -> {result.msoa_code}")
            else:
                click.echo("No similar localities found")
    
    except Exception as e:
        click.echo(f"Error mapping locality: {e}")

@genai.command()
@click.option('--query', '-q', required=True, help='Search query for localities')
def search_localities(query):
    """Search for localities matching a query"""
    click.echo(f"Searching localities for: {query}")
    
    try:
        mapper = LocalityMapper()
        results = mapper.search_localities(query)
        
        if results:
            click.echo(f"\nFound {len(results)} localities:")
            for result in results:
                click.echo(f"  - {result.name} ({result.type})")
                click.echo(f"    MSOA: {result.msoa_code}")
                click.echo(f"    Local Authority: {result.local_authority}")
                click.echo(f"    Confidence: {result.confidence:.2f}")
                click.echo("")
        else:
            click.echo(f"No localities found for: {query}")
    
    except Exception as e:
        click.echo(f"Error searching localities: {e}")

@genai.command()
@click.option('--msoa-code', '-m', required=True, help='MSOA code to validate')
def validate_msoa(msoa_code):
    """Validate an MSOA code"""
    try:
        mapper = LocalityMapper()
        is_valid = mapper.validate_msoa_code(msoa_code)
        
        if is_valid:
            msoa_info = mapper.get_msoa_info(msoa_code)
            click.echo(f"✅ MSOA code {msoa_code} is valid")
            click.echo(f"Local Authority: {msoa_info['la']}")
            click.echo(f"Region: {msoa_info['region']}")
        else:
            click.echo(f"❌ MSOA code {msoa_code} is not valid")
            click.echo("Available MSOA codes:")
            all_codes = mapper.get_all_msoa_codes()
            for code in all_codes[:10]:  # Show first 10
                click.echo(f"  - {code}")
            if len(all_codes) > 10:
                click.echo(f"  ... and {len(all_codes) - 10} more")
    
    except Exception as e:
        click.echo(f"Error validating MSOA code: {e}")

@genai.command()
@click.option('--file', '-f', required=True, help='File containing multiple texts to analyze')
@click.option('--source', '-s', default='batch', help='Source identifier for the batch')
@click.option('--output', '-o', default='json', help='Output format: json, csv, summary')
def batch_analyze(file, source, output):
    """Analyze multiple texts from a file"""
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines or other delimiters
        texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
        if not texts:
            click.echo("No texts found in file")
            return
        
        click.echo(f"Analyzing {len(texts)} texts...")
        
        analyzer = GenAITextAnalyzer()
        text_tuples = [(text, f"{source}_{i+1}", f"batch_{i+1}") for i, text in enumerate(texts)]
        results = analyzer.analyze_multiple_texts(text_tuples)
        
        if output == 'json':
            click.echo(analyzer.export_results(results, 'json'))
        elif output == 'csv':
            click.echo(analyzer.export_results(results, 'csv'))
        elif output == 'summary':
            click.echo(analyzer.export_results(results, 'summary'))
        else:
            click.echo(f"Batch analysis completed. Processed {len(results)} texts.")
            total_issues = sum(result.total_issues for result in results)
            total_critical = sum(result.critical_issues for result in results)
            click.echo(f"Total issues found: {total_issues} (Critical: {total_critical})")
    
    except Exception as e:
        click.echo(f"Error during batch analysis: {e}")

@genai.command()
@click.option('--text1', '-t1', required=True, help='First text for similarity comparison')
@click.option('--text2', '-t2', required=True, help='Second text for similarity comparison')
def similarity(text1, text2):
    """Calculate similarity between two texts using embeddings"""
    click.echo("Calculating text similarity...")
    
    try:
        analyzer = GenAITextAnalyzer()
        similarity_score = analyzer.calculate_text_similarity(text1, text2)
        
        click.echo(f"\nText Similarity Analysis:")
        click.echo(f"Text 1: {text1[:100]}...")
        click.echo(f"Text 2: {text2[:100]}...")
        click.echo(f"Similarity Score: {similarity_score:.4f}")
        
        if similarity_score > 0.8:
            click.echo("✅ Very similar texts")
        elif similarity_score > 0.6:
            click.echo("🟡 Moderately similar texts")
        elif similarity_score > 0.4:
            click.echo("🟠 Somewhat similar texts")
        else:
            click.echo("❌ Dissimilar texts")
    
    except Exception as e:
        click.echo(f"Error calculating similarity: {e}")

@genai.command()
@click.option('--text', '-t', required=True, help='Text to generate embedding for')
@click.option('--output', '-o', default='console', help='Output format: console, json')
def embed(text, output):
    """Generate embedding for a text"""
    click.echo("Generating embedding...")
    
    try:
        analyzer = GenAITextAnalyzer()
        embedding = analyzer.generate_single_embedding(text)
        
        if output == 'json':
            click.echo(json.dumps({
                "text": text,
                "embedding": embedding,
                "model": analyzer.embedding_model,
                "dimensions": len(embedding)
            }, indent=2))
        else:
            click.echo(f"\nEmbedding Analysis:")
            click.echo(f"Text: {text}")
            click.echo(f"Model: {analyzer.embedding_model}")
            click.echo(f"Dimensions: {len(embedding)}")
            click.echo(f"First 10 values: {embedding[:10]}")
    
    except Exception as e:
        click.echo(f"Error generating embedding: {e}")

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

@cli.group()
def population():
    """Population data analysis commands"""
    pass

@population.command()
@click.option('--msoa-code', help='MSOA code to get population data for')
@click.option('--summary', is_flag=True, help='Show overall population summary')
@click.option('--top', type=int, default=10, help='Show top N most populated MSOAs')
def analyze(msoa_code, summary, top):
    """Analyze population data"""
    connector = UnifiedDataConnector()
    
    if summary:
        click.echo("📊 Population Summary")
        click.echo("=" * 50)
        summary_data = connector.get_population_summary()
        
        if summary_data:
            click.echo(f"Total Population: {summary_data.get('total_population', 0):,}")
            click.echo(f"Total MSOAs: {summary_data.get('total_msoas', 0)}")
            click.echo(f"Average Population per MSOA: {summary_data.get('average_population_per_msoa', 0):.0f}")
            click.echo(f"Min Population: {summary_data.get('min_population', 0):,}")
            click.echo(f"Max Population: {summary_data.get('max_population', 0):,}")
            
            age_groups = summary_data.get('age_groups', {})
            if age_groups:
                click.echo(f"\nAge Group Distribution:")
                for age_group, count in age_groups.items():
                    percentage = (count / summary_data.get('total_population', 1)) * 100
                    click.echo(f"  {age_group}: {count:,} ({percentage:.1f}%)")
        else:
            click.echo("❌ No population data available")
    
    elif msoa_code:
        click.echo(f"👥 Population Data for MSOA: {msoa_code}")
        click.echo("=" * 50)
        
        pop_data = connector.get_population_data(msoa_code)
        if pop_data:
            click.echo(f"MSOA Name: {pop_data.get('msoa_name', 'Unknown')}")
            click.echo(f"Total Population: {pop_data.get('total_population', 0):,}")
            
            # Demographic analysis
            demo_analysis = connector.get_demographic_analysis(msoa_code)
            if demo_analysis:
                gender = demo_analysis.get('gender_distribution', {})
                click.echo(f"Gender Distribution:")
                click.echo(f"  Female: {gender.get('female', 0):,} ({gender.get('female_percentage', 0):.1f}%)")
                click.echo(f"  Male: {gender.get('male', 0):,} ({gender.get('male_percentage', 0):.1f}%)")
                
                age_groups = demo_analysis.get('age_groups', {})
                if age_groups:
                    click.echo(f"\nAge Group Distribution:")
                    for age_group, count in age_groups.items():
                        percentage = (count / pop_data.get('total_population', 1)) * 100
                        click.echo(f"  {age_group}: {count:,} ({percentage:.1f}%)")
        else:
            click.echo("❌ No population data found for this MSOA")
    
    else:
        # Show top populated MSOAs
        click.echo(f"🏆 Top {top} Most Populated MSOAs")
        click.echo("=" * 50)
        
        top_msoas = connector.get_top_populated_msoas(top)
        if top_msoas:
            for msoa in top_msoas:
                click.echo(f"{msoa['rank']:2d}. {msoa['msoa_name']} ({msoa['msoa_code']})")
                click.echo(f"    Population: {msoa['total_population']:,}")
        else:
            click.echo("❌ No population data available")

@population.command()
def refresh_cache():
    """Refresh the population data cache"""
    click.echo("🔄 Refreshing population data cache...")
    
    # Create connector without auto-loading data
    connector = UnifiedDataConnector(auto_load=False)
    
    # Force refresh the cache
    connector.refresh_population_cache()

@cli.group()
def early_warning():
    """Early warning system commands"""
    pass

@early_warning.command()
@click.option('--use-real', is_flag=True, default=True, help='Use real data (default: True)')
@click.option('--use-dummy', is_flag=True, help='Use dummy data instead of real data')
@click.option('--output', help='Output file for results (JSON format)')
def analyze(use_real, use_dummy, output):
    """Run early warning analysis"""
    from src.early_warning_system import EarlyWarningSystem
    
    click.echo("🚨 Early Warning System Analysis")
    click.echo("=" * 50)
    
    # Determine data source
    use_real_data = use_real and not use_dummy
    
    # Initialize early warning system
    ews = EarlyWarningSystem()
    
    # Load data
    click.echo(f"📊 Loading {'real' if use_real_data else 'dummy'} data...")
    data = ews.load_data(use_real=use_real_data)
    
    # Run full analysis
    click.echo("🔄 Running analysis...")
    results = ews.run_full_analysis(data)
    
    # Display summary
    summary = results['summary']
    click.echo(f"\n📈 Analysis Summary:")
    click.echo(f"Total Areas Analyzed: {summary['total_areas']}")
    click.echo(f"Critical Risk Areas: {summary['critical_risk_areas']}")
    click.echo(f"High Risk Areas: {summary['high_risk_areas']}")
    click.echo(f"Anomalous Areas: {summary['anomalous_areas']}")
    click.echo(f"Total Alerts: {summary['total_alerts']}")
    click.echo(f"Average Risk Score: {summary['average_risk_score']:.3f}")
    
    # Display risk distribution
    click.echo(f"\n📊 Risk Distribution:")
    for risk_level, count in summary['risk_distribution'].items():
        click.echo(f"  {risk_level}: {count}")
    
    # Display top alerts
    alerts = results['alerts']
    if alerts:
        click.echo(f"\n🚨 Top Alerts:")
        for i, alert in enumerate(alerts[:5], 1):
            click.echo(f"{i}. {alert['message']} (Priority: {alert['priority']})")
    
    # Save results if output file specified
    if output:
        import json
        # Convert DataFrame to dict for JSON serialization
        results_copy = results.copy()
        results_copy['data'] = results['data'].to_dict('records')
        
        with open(output, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        click.echo(f"\n💾 Results saved to {output}")

@early_warning.command()
@click.argument('msoa_code')
def risk_factors(msoa_code):
    """Get detailed risk factors for a specific MSOA"""
    from src.early_warning_system import EarlyWarningSystem
    
    click.echo(f"🔍 Risk Factors Analysis for MSOA: {msoa_code}")
    click.echo("=" * 50)
    
    # Initialize early warning system
    ews = EarlyWarningSystem()
    
    # Load data and run analysis
    data = ews.load_data(use_real=True)
    results = ews.run_full_analysis(data)
    
    # Get risk factors for specific MSOA
    risk_factors = ews.get_risk_factors(results['data'], msoa_code)
    
    if 'error' in risk_factors:
        click.echo(f"❌ {risk_factors['error']}")
        return
    
    # Display results
    click.echo(f"MSOA Code: {risk_factors['msoa_code']}")
    click.echo(f"Local Authority: {risk_factors['local_authority']}")
    click.echo(f"Overall Risk Score: {risk_factors['overall_risk_score']:.3f}")
    click.echo(f"Risk Level: {risk_factors['risk_level']}")
    
    click.echo(f"\n🚨 Top Risk Factors:")
    for factor, score in risk_factors['top_risk_factors']:
        click.echo(f"  {factor}: {score:.3f}")
    
    click.echo(f"\n🛡️ Protective Factors:")
    for factor, score in risk_factors['protective_factors']:
        click.echo(f"  {factor}: {score:.3f}")
    
    click.echo(f"\n💡 Recommendations:")
    for rec in risk_factors['recommendations']:
        click.echo(f"  • {rec}")

@cli.group()
def community_survey():
    """Community Life Survey commands"""
    pass

@community_survey.command()
def analyze():
    """Analyze Community Life Survey data structure"""
    from src.community_life_survey_connector import CommunityLifeSurveyConnector
    
    click.echo("📊 Community Life Survey Analysis")
    click.echo("=" * 50)
    
    connector = CommunityLifeSurveyConnector()
    
    # Load all sheets
    click.echo("🔄 Loading all sheets...")
    sheets = connector.load_all_sheets()
    
    if not sheets:
        click.echo("❌ No sheets could be loaded")
        return
    
    # Analyze first few sheets
    click.echo(f"\n🔍 Analyzing first 3 sheets:")
    for i, sheet_name in enumerate(list(sheets.keys())[:3]):
        click.echo(f"\nSheet {i+1}: {sheet_name}")
        analysis = connector.analyze_sheet_structure(sheet_name)
        click.echo(f"  Data start row: {analysis.get('data_start_row', 'Not found')}")
        click.echo(f"  Question: {analysis.get('question', 'Not found')}")
        click.echo(f"  Sample LADs: {analysis.get('sample_lads', [])[:3]}")

@community_survey.command()
def process():
    """Process all Community Life Survey sheets"""
    from src.community_life_survey_connector import CommunityLifeSurveyConnector
    
    click.echo("🔄 Processing Community Life Survey Data")
    click.echo("=" * 50)
    
    connector = CommunityLifeSurveyConnector()
    
    # Process all sheets
    click.echo("📊 Processing all sheets...")
    processed_data = connector.process_all_sheets()
    
    if processed_data.empty:
        click.echo("❌ No data could be processed")
        return
    
    # Get summary
    summary = connector.get_question_summary()
    click.echo(f"\n📈 Processing Summary:")
    click.echo(f"  Total questions: {summary.get('total_questions', 0)}")
    click.echo(f"  Total sheets: {summary.get('total_sheets', 0)}")
    click.echo(f"  Total responses: {summary.get('total_responses', 0)}")
    
    # Export processed data
    click.echo("\n💾 Exporting processed data...")
    connector.export_processed_data()
    
    click.echo("✅ Community Life Survey data processing complete!")

@community_survey.command()
@click.argument('lad_name')
def lad_data(lad_name):
    """Get all data for a specific Local Authority District"""
    from src.community_life_survey_connector import CommunityLifeSurveyConnector
    
    click.echo(f"🏛️ Community Life Survey Data for: {lad_name}")
    click.echo("=" * 50)
    
    connector = CommunityLifeSurveyConnector()
    
    # Load and process data
    connector.load_all_sheets()
    processed_data = connector.process_all_sheets()
    
    if processed_data.empty:
        click.echo("❌ No data available")
        return
    
    # Get LAD data
    lad_data = connector.get_lad_data(lad_name)
    
    if lad_data.empty:
        click.echo(f"❌ No data found for LAD: {lad_name}")
        click.echo("Available LADs:")
        available_lads = processed_data.iloc[:, 1].dropna().unique()[:10]
        for lad in available_lads:
            click.echo(f"  • {lad}")
        return
    
    click.echo(f"📊 Found {len(lad_data)} responses for {lad_name}")
    
    # Show sample data
    click.echo(f"\nSample data:")
    for i, (_, row) in enumerate(lad_data.head(5).iterrows()):
        question = row.get('question', 'Unknown')
        click.echo(f"  {i+1}. {question}")

@cli.group()
def survey():
    """Community Life Survey commands"""
    pass

@survey.command()
def summary():
    """Get Community Life Survey summary"""
    from src.unified_data_connector import UnifiedDataConnector
    
    click.echo("📋 Community Life Survey Summary")
    click.echo("=" * 50)
    
    connector = UnifiedDataConnector()
    summary = connector.get_community_survey_summary()
    
    if not summary:
        click.echo("❌ No Community Life Survey data available")
        return
    
    click.echo(f"Total Questions: {summary.get('total_questions', 0)}")
    click.echo(f"Total Sheets: {summary.get('total_sheets', 0)}")
    click.echo(f"Total Responses: {summary.get('total_responses', 0)}")
    click.echo(f"Unique Local Authorities: {summary.get('unique_lads', 0)}")

@survey.command()
@click.argument('lad_name')
def lad_data(lad_name):
    """Get Community Life Survey data for a specific LAD"""
    from src.unified_data_connector import UnifiedDataConnector
    
    click.echo(f"🏛️ Community Life Survey Data for: {lad_name}")
    click.echo("=" * 50)
    
    connector = UnifiedDataConnector()
    lad_data = connector.get_lad_survey_data(lad_name)
    
    if lad_data.empty:
        click.echo(f"❌ No data found for LAD: {lad_name}")
        return
    
    click.echo(f"📊 Found {len(lad_data)} responses for {lad_name}")
    
    # Show sample data
    click.echo(f"\nSample questions:")
    for i, (_, row) in enumerate(lad_data.head(5).iterrows()):
        question = row.get('question', 'Unknown')
        click.echo(f"  {i+1}. {question}")

@survey.command()
@click.argument('question')
def question_data(question):
    """Get Community Life Survey data for a specific question"""
    from src.unified_data_connector import UnifiedDataConnector
    
    click.echo(f"❓ Community Life Survey Data for Question: {question}")
    click.echo("=" * 50)
    
    connector = UnifiedDataConnector()
    question_data = connector.get_survey_question_data(question)
    
    if question_data.empty:
        click.echo(f"❌ No data found for question: {question}")
        return
    
    click.echo(f"📊 Found {len(question_data)} responses for this question")
    
    # Show sample LADs
    lad_column = question_data.columns[1]  # Column B should be LAD names
    sample_lads = question_data[lad_column].head(10).tolist()
    click.echo(f"\nSample Local Authorities:")
    for i, lad in enumerate(sample_lads, 1):
        click.echo(f"  {i}. {lad}")

@survey.command()
def refresh():
    """Refresh Community Life Survey data to pick up updated question text"""
    from src.unified_data_connector import UnifiedDataConnector
    
    click.echo("🔄 Refreshing Community Life Survey Data")
    click.echo("=" * 50)
    
    connector = UnifiedDataConnector()
    connector.refresh_community_survey_data()
    
    # Show updated questions
    questions = connector.get_all_survey_questions()
    click.echo(f"\n📋 Updated Questions ({len(questions)} total):")
    for i, question in enumerate(questions[:10], 1):  # Show first 10
        click.echo(f"  {i}. {question}")
    
    if len(questions) > 10:
        click.echo(f"  ... and {len(questions) - 10} more questions")

@survey.command()
def clean():
    """Clean Community Life Survey data by removing header rows"""
    import subprocess
    import sys
    
    click.echo("🧹 Cleaning Community Life Survey Data")
    click.echo("=" * 50)
    
    try:
        # Run the cleaning script
        result = subprocess.run([sys.executable, "clean_community_survey.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo("✅ Data cleaning completed successfully!")
            click.echo("\nOutput:")
            click.echo(result.stdout)
        else:
            click.echo("❌ Data cleaning failed!")
            click.echo("\nError:")
            click.echo(result.stderr)
            
    except Exception as e:
        click.echo(f"❌ Error running cleaning script: {e}")

if __name__ == '__main__':
    cli()
