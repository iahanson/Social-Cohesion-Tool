#!/usr/bin/env python3
"""
Example usage of the Social Cohesion Monitoring System
Demonstrates all major components with sample data
"""

from src.data_aggregator import MSOADataAggregator
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator

def main():
    print("🏘️ Social Cohesion Monitoring System - Example Usage")
    print("=" * 60)
    
    # 1. Basic MSOA Lookup (Original functionality)
    print("\n📍 1. MSOA Data Lookup")
    print("-" * 30)
    
    aggregator = MSOADataAggregator()
    example_postcodes = ["SW1A 1AA", "M1 1AA", "B1 1AA"]
    
    for postcode in example_postcodes:
        msoa_data = aggregator.get_msoa_by_postcode(postcode)
        if msoa_data:
            print(f"{postcode}: {msoa_data.get('msoa_code', 'N/A')} - {msoa_data.get('local_authority', 'N/A')}")
    
    # 2. Early Warning System
    print("\n🚨 2. Early Warning System")
    print("-" * 30)
    
    ew_system = EarlyWarningSystem()
    ew_results = ew_system.run_full_analysis()
    print(f"Analyzed {ew_results['summary']['total_areas']} areas")
    print(f"Critical risk areas: {ew_results['summary']['critical_risk_areas']}")
    print(f"High risk areas: {ew_results['summary']['high_risk_areas']}")
    print(f"Anomalous areas: {ew_results['summary']['anomalous_areas']}")
    
    if ew_results['alerts']:
        print(f"\nGenerated {len(ew_results['alerts'])} alerts:")
        for alert in ew_results['alerts'][:3]:  # Show first 3
            print(f"  - {alert['type']}: {alert['message']}")
    
    # 3. Sentiment & Trust Mapping
    print("\n🗺️ 3. Sentiment & Trust Mapping")
    print("-" * 30)
    
    sm_system = SentimentMapping()
    sm_results = sm_system.run_full_analysis()
    print(f"Average trust score: {sm_results['summary']['average_trust_score']:.1f}/10")
    print(f"Average cohesion score: {sm_results['summary']['average_cohesion_score']:.1f}/10")
    print(f"Average sentiment score: {sm_results['summary']['average_sentiment_score']:.1f}/10")
    
    # 4. Intervention Recommendations
    print("\n💡 4. Intervention Recommendations")
    print("-" * 30)
    
    int_tool = InterventionTool()
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
    
    int_results = int_tool.run_full_analysis(target_area)
    print(f"Database contains {int_results['summary']['total_cases_in_database']} historical cases")
    print(f"Top intervention type: {int_results['summary']['top_performing_intervention']}")
    
    if int_results['recommendations']:
        print(f"\nTop 3 recommendations:")
        for i, rec in enumerate(int_results['recommendations'][:3], 1):
            print(f"  {i}. {rec['intervention_type']} (Success: {rec['expected_success_score']:.2f})")
    
    # 5. Community Engagement Simulator
    print("\n🎯 5. Community Engagement Simulator")
    print("-" * 30)
    
    simulator = EngagementSimulator()
    interventions = {
        'community_events_investment': 80,
        'youth_programs_investment': 60,
        'cultural_exchange_investment': 70,
        'sports_recreation_investment': 50
    }
    
    sim_result = simulator.simulate_intervention_impact(target_area, interventions)
    print(f"Total investment: £{sim_result['investment_summary']['total_investment']:,.0f}")
    print(f"Expected trust improvement: {sim_result['improvements']['trust_improvement']:.2f}")
    print(f"Expected cohesion improvement: {sim_result['improvements']['cohesion_improvement']:.2f}")
    print(f"Expected sentiment improvement: {sim_result['improvements']['sentiment_improvement']:.2f}")
    print(f"Overall improvement: {sim_result['improvements']['overall_improvement']:.2f}")
    print(f"Cost effectiveness: {sim_result['investment_summary']['cost_effectiveness']:.3f}")
    
    # 6. Summary
    print("\n📊 System Summary")
    print("-" * 30)
    print("✅ All components working successfully!")
    print("✅ Sample data generated and processed")
    print("✅ Risk analysis completed")
    print("✅ Intervention recommendations generated")
    print("✅ Impact simulation completed")
    print("\n🚀 Ready for real data integration!")
    print("\nNext steps:")
    print("  - Run 'python main.py dashboard' for interactive web interface")
    print("  - Run 'python main.py status' to check system configuration")
    print("  - Configure .env file for email/SMS alerts")

if __name__ == "__main__":
    main()
