"""
Main Data Aggregator
Combines data from multiple sources for MSOA analysis
"""

import json
import csv
import sys
from typing import Optional, Dict, Any
from .msoa_search import MSOASearch
from .imd_connector import IMDConnector

class MSOADataAggregator:
    """Main class for aggregating MSOA data from multiple sources"""
    
    def __init__(self):
        self.msoa_search = MSOASearch()
        self.imd_connector = IMDConnector()
    
    def get_msoa_by_postcode(self, postcode: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive MSOA data by postcode
        
        Args:
            postcode: UK postcode
            
        Returns:
            Dictionary containing all available MSOA data
        """
        # Get basic MSOA information
        msoa_info = self.msoa_search.postcode_to_msoa(postcode)
        if not msoa_info:
            return None
        
        # Get IMD data if MSOA code is available
        if msoa_info.get('msoa_code'):
            imd_data = self.imd_connector.get_imd_for_msoa(msoa_info['msoa_code'])
            if imd_data:
                msoa_info['imd_data'] = imd_data
        
        return msoa_info
    
    def get_msoa_by_code(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive MSOA data by MSOA code
        
        Args:
            msoa_code: MSOA code
            
        Returns:
            Dictionary containing all available MSOA data
        """
        # Validate MSOA code
        if not self.msoa_search.validate_msoa_code(msoa_code):
            return None
        
        # Get basic MSOA information
        msoa_info = self.msoa_search.get_msoa_details(msoa_code)
        if not msoa_info:
            return None
        
        # Get IMD data
        imd_data = self.imd_connector.get_imd_for_msoa(msoa_code)
        if imd_data:
            msoa_info['imd_data'] = imd_data
        
        return msoa_info
    
    def display_results(self, msoa_data: Dict[str, Any], output_format: str = 'console'):
        """
        Display MSOA data in the specified format
        
        Args:
            msoa_data: Dictionary containing MSOA data
            output_format: Output format ('console', 'json', 'csv')
        """
        if output_format == 'json':
            print(json.dumps(msoa_data, indent=2))
        elif output_format == 'csv':
            self._output_csv(msoa_data)
        else:  # console
            self._output_console(msoa_data)
    
    def _output_console(self, msoa_data: Dict[str, Any]):
        """Display data in console format"""
        print("\n" + "="*60)
        print("UK MSOA DATA REPORT")
        print("="*60)
        
        # Basic MSOA information
        print(f"\n📍 Location Information:")
        print(f"   Postcode: {msoa_data.get('postcode', 'N/A')}")
        print(f"   MSOA Code: {msoa_data.get('msoa_code', 'N/A')}")
        print(f"   MSOA Name: {msoa_data.get('msoa_name', 'N/A')}")
        print(f"   Local Authority: {msoa_data.get('local_authority', 'N/A')}")
        print(f"   Region: {msoa_data.get('region', 'N/A')}")
        print(f"   Country: {msoa_data.get('country', 'N/A')}")
        
        if msoa_data.get('longitude') and msoa_data.get('latitude'):
            print(f"   Coordinates: {msoa_data['latitude']:.6f}, {msoa_data['longitude']:.6f}")
        
        # IMD data
        if msoa_data.get('imd_data'):
            imd = msoa_data['imd_data']
            print(f"\n📊 Indices of Multiple Deprivation (2019):")
            print(f"   IMD Rank: {imd.get('imd_rank', 'N/A'):.0f}")
            print(f"   IMD Decile: {imd.get('imd_decile', 'N/A'):.1f}")
            print(f"   Total LSOAs in MSOA: {imd.get('total_lsoas', 'N/A')}")
            
            print(f"\n   Domain Rankings (1 = most deprived):")
            print(f"   • Income: {imd.get('income_rank', 'N/A'):.0f}")
            print(f"   • Employment: {imd.get('employment_rank', 'N/A'):.0f}")
            print(f"   • Education: {imd.get('education_rank', 'N/A'):.0f}")
            print(f"   • Health: {imd.get('health_rank', 'N/A'):.0f}")
            print(f"   • Crime: {imd.get('crime_rank', 'N/A'):.0f}")
            print(f"   • Housing Barriers: {imd.get('barriers_rank', 'N/A'):.0f}")
            print(f"   • Living Environment: {imd.get('living_rank', 'N/A'):.0f}")
        
        print("\n" + "="*60)
    
    def _output_csv(self, msoa_data: Dict[str, Any]):
        """Output data in CSV format"""
        # Flatten the data for CSV output
        flat_data = {}
        
        # Basic info
        for key in ['postcode', 'msoa_code', 'msoa_name', 'local_authority', 'region', 'country', 'longitude', 'latitude']:
            flat_data[key] = msoa_data.get(key, '')
        
        # IMD data
        if msoa_data.get('imd_data'):
            imd = msoa_data['imd_data']
            for key in ['imd_rank', 'imd_decile', 'total_lsoas', 'income_rank', 'employment_rank', 
                       'education_rank', 'health_rank', 'crime_rank', 'barriers_rank', 'living_rank']:
                flat_data[f'imd_{key}'] = imd.get(key, '')
        
        # Write CSV
        if flat_data:
            writer = csv.DictWriter(sys.stdout, fieldnames=flat_data.keys())
            writer.writeheader()
            writer.writerow(flat_data)
    
    def get_available_sources(self) -> Dict[str, str]:
        """Get list of available data sources"""
        return {
            'imd_2019': 'English Indices of Multiple Deprivation 2019',
            'ons_postcode': 'ONS Postcode Directory (via postcodes.io)',
            'msoa_boundaries': 'MSOA Boundaries and Names'
        }
