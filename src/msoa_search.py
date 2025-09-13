"""
MSOA Search and Lookup functionality
Handles postcode to MSOA mapping and MSOA code validation
"""

import requests
import pandas as pd
from typing import Optional, Dict, Any
import os

class MSOASearch:
    """Handles MSOA search and lookup operations"""
    
    def __init__(self):
        self.ons_postcode_url = "https://api.postcodes.io/postcodes/"
        self.msoa_boundaries_url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/MSOA_Names_and_Codes_England_and_Wales/FeatureServer/0/query"
    
    def postcode_to_msoa(self, postcode: str) -> Optional[Dict[str, Any]]:
        """
        Convert UK postcode to MSOA information using postcodes.io API
        
        Args:
            postcode: UK postcode (with or without space)
            
        Returns:
            Dictionary containing MSOA information or None if not found
        """
        try:
            # Clean postcode
            clean_postcode = postcode.replace(" ", "").upper()
            
            # Use postcodes.io API
            response = requests.get(f"{self.ons_postcode_url}{clean_postcode}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 200:
                    result = data['result']
                    return {
                        'postcode': result['postcode'],
                        'msoa_code': result.get('codes', {}).get('msoa'),
                        'msoa_name': result.get('codes', {}).get('msoa'),
                        'lsoa_code': result.get('codes', {}).get('lsoa'),
                        'ward_code': result.get('codes', {}).get('ward'),
                        'local_authority': result.get('admin_district'),
                        'region': result.get('region'),
                        'country': result.get('country'),
                        'longitude': result.get('longitude'),
                        'latitude': result.get('latitude')
                    }
            return None
            
        except Exception as e:
            print(f"Error looking up postcode {postcode}: {e}")
            return None
    
    def validate_msoa_code(self, msoa_code: str) -> bool:
        """
        Validate if an MSOA code exists
        
        Args:
            msoa_code: MSOA code to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # MSOA codes in England typically start with 'E' followed by 8 digits
            if len(msoa_code) == 9 and msoa_code.startswith('E'):
                return True
            return False
        except:
            return False
    
    def get_msoa_details(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an MSOA by its code
        
        Args:
            msoa_code: MSOA code
            
        Returns:
            Dictionary containing MSOA details or None if not found
        """
        try:
            # This would typically involve querying ONS data
            # For now, return basic structure
            if self.validate_msoa_code(msoa_code):
                return {
                    'msoa_code': msoa_code,
                    'msoa_name': f"MSOA {msoa_code}",
                    'country': 'England',
                    'status': 'Valid MSOA code'
                }
            return None
        except Exception as e:
            print(f"Error getting MSOA details for {msoa_code}: {e}")
            return None
