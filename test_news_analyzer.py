#!/usr/bin/env python3
"""
Standalone News Analyzer Test Script
Analyzes local news for a given location and saves results to CSV
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.local_news_analyzer import LocalNewsAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Test News Analyzer for a specific location')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the location')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the location')
    parser.add_argument('--area', type=str, default='Test Area', help='Name of the area (optional)')
    parser.add_argument('--output', type=str, default='news_analysis_results.csv', help='Output CSV filename')
    parser.add_argument('--json', action='store_true', help='Also save detailed results as JSON')

    args = parser.parse_args()

    print(f"🎯 Testing News Analyzer for location: {args.area}")
    print(f"📍 Coordinates: ({args.lat}, {args.lon})")
    print("=" * 60)

    try:
        # Initialize analyzer
        print("🔧 Initializing news analyzer...")
        analyzer = LocalNewsAnalyzer()

        # Run analysis
        print("🔍 Analyzing local news...")
        results = analyzer.analyze_location_comprehensive(args.lat, args.lon, args.area)

        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return

        # Print summary
        print(f"✅ Analysis completed!")
        print(f"📰 Found {results['total_articles']} articles from {len(results['sources_analyzed'])} sources")

        # Prepare data for CSV
        csv_data = []

        # Add summary row
        summary_row = {
            'Type': 'SUMMARY',
            'Area': results['area_name'],
            'Latitude': results['coordinates'][0],
            'Longitude': results['coordinates'][1],
            'Total_Articles': results['total_articles'],
            'Sources_Count': len(results['sources_analyzed']),
            'Analysis_Date': results['scraped_at'],
            'Title': '',
            'Source': '',
            'URL': '',
            'Traffic_Score': '',
            'Sentiment_Score': results['aggregate_analysis'].get('sentiment_score', 0),
            'Trust_Score': results['aggregate_analysis'].get('trust_score', 0),
            'Top_Topics': ', '.join(results['aggregate_analysis'].get('top_topics', [])),
            'Confidence': results['aggregate_analysis'].get('confidence', 'Medium')
        }
        csv_data.append(summary_row)

        # Add featured article
        featured = results['most_popular_article']
        featured_row = {
            'Type': 'FEATURED_ARTICLE',
            'Area': results['area_name'],
            'Latitude': results['coordinates'][0],
            'Longitude': results['coordinates'][1],
            'Total_Articles': results['total_articles'],
            'Sources_Count': len(results['sources_analyzed']),
            'Analysis_Date': results['scraped_at'],
            'Title': featured['title'],
            'Source': featured['source'],
            'URL': featured['url'],
            'Traffic_Score': featured.get('traffic_score', 0),
            'Sentiment_Score': results['detailed_analysis'].get('sentiment_score', 0),
            'Trust_Score': results['detailed_analysis'].get('trust_score', 0),
            'Top_Topics': ', '.join(results['detailed_analysis'].get('top_topics', [])),
            'Confidence': results['aggregate_analysis'].get('confidence', 'Medium'),
            'Image_URL': featured.get('image_url', ''),
            'Description': featured.get('description', ''),
            'Content_Preview': featured.get('content_preview', '')
        }
        csv_data.append(featured_row)

        # Add all other articles
        for article in results['all_articles']:
            article_row = {
                'Type': 'ARTICLE',
                'Area': results['area_name'],
                'Latitude': results['coordinates'][0],
                'Longitude': results['coordinates'][1],
                'Total_Articles': results['total_articles'],
                'Sources_Count': len(results['sources_analyzed']),
                'Analysis_Date': results['scraped_at'],
                'Title': article['title'],
                'Source': article['source'],
                'URL': article['url'],
                'Traffic_Score': article.get('traffic_score', 0),
                'Sentiment_Score': '',  # Not calculated for individual articles in bulk
                'Trust_Score': '',
                'Top_Topics': '',
                'Confidence': results['aggregate_analysis'].get('confidence', 'Medium'),
                'Image_URL': '',
                'Description': '',
                'Content_Preview': ''
            }
            csv_data.append(article_row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output, index=False)
        print(f"💾 Results saved to: {args.output}")

        # Save detailed JSON if requested
        if args.json:
            json_filename = args.output.replace('.csv', '.json')
            with open(json_filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Detailed results saved to: {json_filename}")

        # Print key insights
        print("\n📊 KEY INSIGHTS:")
        print("=" * 30)

        # Featured article
        print(f"🏆 Most Popular Article:")
        print(f"   Title: {featured['title'][:80]}...")
        print(f"   Source: {featured['source']}")
        print(f"   Traffic Score: {featured.get('traffic_score', 0):.1f}")

        # Sentiment analysis
        agg = results['aggregate_analysis']
        print(f"\n💭 Overall Sentiment:")
        print(f"   Sentiment Score: {agg.get('sentiment_score', 0):.3f}")
        print(f"   Trust Score: {agg.get('trust_score', 0):.1f}/10")
        print(f"   Confidence: {agg.get('confidence', 'Medium')}")

        # Top topics
        if agg.get('top_topics'):
            print(f"\n🏷️ Top Topics:")
            for i, topic in enumerate(agg['top_topics'][:3], 1):
                print(f"   {i}. {topic.replace('_', ' ').title()}")

        # Sources analyzed
        print(f"\n📰 News Sources:")
        for source in results['sources_analyzed']:
            print(f"   • {source['name']}: {source['articles_found']} articles ({source['distance']:.1f}km away)")

        # AI Analysis if available
        if 'agent_analysis' in results and results['agent_analysis'].get('agent_analysis', False):
            agent = results['agent_analysis']
            print(f"\n🤖 AI Analysis:")
            if agent.get('summary'):
                print(f"   Summary: {agent['summary'][:100]}...")
            if agent.get('main_topics'):
                print(f"   AI Topics: {', '.join([t.replace('_', ' ').title() for t in agent['main_topics'][:3]])}")

        print(f"\n✅ Analysis complete! Check {args.output} for full results.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # If no arguments provided, show example usage
    if len(sys.argv) == 1:
        print("📰 News Analyzer Test Script")
        print("=" * 40)
        print("Usage examples:")
        print()
        print("# Test Glasgow, Scotland")
        print("python test_news_analyzer.py --lat 55.8642 --lon -4.2518 --area 'Glasgow' --output glasgow_news.csv")
        print()
        print("# Test Manchester, England")
        print("python test_news_analyzer.py --lat 53.4808 --lon -2.2426 --area 'Manchester' --output manchester_news.csv")
        print()
        print("# Test London with JSON output")
        print("python test_news_analyzer.py --lat 51.5074 --lon -0.1278 --area 'London' --output london_news.csv --json")
        print()
        print("# Quick test with current directory")
        print("python test_news_analyzer.py --lat 55.8642 --lon -4.2518")
        print()
        print("Available arguments:")
        print("  --lat LATITUDE    (required) Latitude coordinate")
        print("  --lon LONGITUDE   (required) Longitude coordinate")
        print("  --area AREA_NAME  (optional) Name for the area")
        print("  --output FILENAME (optional) CSV output filename")
        print("  --json           (optional) Also save detailed JSON results")
        sys.exit(0)

    main()