"""
LSOA to MSOA Mapping Module
Handles the mapping between Lower Layer Super Output Areas (LSOA) and 
Middle Layer Super Output Areas (MSOA) for UK geographic data.
"""

import pandas as pd
import requests
import json
from typing import Dict, Optional, List
import os
from datetime import datetime, timedelta

class LSOAMSOAMapper:
    """Maps LSOA codes to MSOA codes using ONS lookup tables"""
    
    def __init__(self):
        self.lsoa_msoa_mapping = {}
        self.msoa_lsoa_mapping = {}
        self.cache_file = "data/lsoa_msoa_mapping.json"
        self.cache_expiry_hours = 24 * 7  # 1 week
        self.census_lookup_file = "data/Census21 areaLookupTable.xlsx"
        
    def load_mapping_data(self) -> bool:
        """Load LSOA to MSOA mapping data"""
        try:
            # Try to load from cache first
            if self._load_from_cache():
                print("✅ LSOA-MSOA mapping loaded from cache")
                return True
            
            # Try to load from Census lookup table
            if self._load_from_census_lookup():
                self._save_to_cache()
                print("✅ LSOA-MSOA mapping loaded from Census 2021 lookup table")
                return True
            
            # If Census lookup fails, try ONS API
            if self._fetch_from_ons():
                self._save_to_cache()
                print("✅ LSOA-MSOA mapping fetched from ONS API")
                return True
            
            # Fallback to built-in mapping for common areas
            self._load_fallback_mapping()
            print("⚠️ Using fallback LSOA-MSOA mapping")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LSOA-MSOA mapping: {e}")
            return False
    
    def _load_from_cache(self) -> bool:
        """Load mapping from local cache if it exists and is fresh"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            # Check if cache is fresh
            cache_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
            if datetime.now() - cache_time > timedelta(hours=self.cache_expiry_hours):
                return False
            
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.lsoa_msoa_mapping = data.get('lsoa_to_msoa', {})
                self.msoa_lsoa_mapping = data.get('msoa_to_lsoa', {})
            
            return len(self.lsoa_msoa_mapping) > 0
            
        except Exception:
            return False
    
    def _save_to_cache(self):
        """Save mapping to local cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                'lsoa_to_msoa': self.lsoa_msoa_mapping,
                'msoa_to_lsoa': self.msoa_lsoa_mapping,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    def _fetch_from_ons(self) -> bool:
        """Fetch LSOA to MSOA mapping from ONS API"""
        try:
            # ONS LSOA to MSOA lookup table URL
            url = "https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2011-to-msoa-2011-to-local-authority-district-2011-lookup-in-england-and-wales/explore"
            
            # Alternative: Use the direct API endpoint
            api_url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LSOA_2011_to_MSOA_2011_to_LAD_2011_Lookup_in_England_and_Wales/FeatureServer/0/query"
            
            params = {
                'where': '1=1',
                'outFields': 'LSOA11CD,MSOA11CD,LAD11CD,LSOA11NM,MSOA11NM,LAD11NM',
                'f': 'json',
                'outSR': '4326'
            }
            
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            if not features:
                return False
            
            # Process the data
            for feature in features:
                attrs = feature.get('attributes', {})
                lsoa_code = attrs.get('LSOA11CD')
                msoa_code = attrs.get('MSOA11CD')
                
                if lsoa_code and msoa_code:
                    self.lsoa_msoa_mapping[lsoa_code] = msoa_code
                    
                    # Build reverse mapping
                    if msoa_code not in self.msoa_lsoa_mapping:
                        self.msoa_lsoa_mapping[msoa_code] = []
                    self.msoa_lsoa_mapping[msoa_code].append(lsoa_code)
            
            return len(self.lsoa_msoa_mapping) > 0
            
        except Exception as e:
            print(f"⚠️ Could not fetch from ONS: {e}")
            return False
    
    def _load_from_census_lookup(self) -> bool:
        """Load LSOA to MSOA mapping from Census 2021 lookup table"""
        try:
            if not os.path.exists(self.census_lookup_file):
                print(f"⚠️ Census lookup file not found: {self.census_lookup_file}")
                return False
            
            # Read the Census lookup table
            df = pd.read_excel(self.census_lookup_file)
            
            # Try different possible column names for LSOA and MSOA codes
            lsoa_col = None
            msoa_col = None
            
            # Common column name variations (including lowercase)
            lsoa_candidates = ['LSOA21CD', 'lsoa21cd', 'LSOA_CD', 'LSOA_CODE', 'LSOA11CD', 'LSOA_2011_CD']
            msoa_candidates = ['MSOA21CD', 'msoa21cd', 'MSOA_CD', 'MSOA_CODE', 'MSOA11CD', 'MSOA_2011_CD']
            
            for col in df.columns:
                if col in lsoa_candidates:
                    lsoa_col = col
                if col in msoa_candidates:
                    msoa_col = col
            
            if not lsoa_col or not msoa_col:
                print(f"⚠️ Could not find LSOA/MSOA columns in Census lookup table")
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Looking for LSOA columns: {lsoa_candidates}")
                print(f"Looking for MSOA columns: {msoa_candidates}")
                return False
            
            print(f"Using columns: {lsoa_col} -> {msoa_col}")
            
            # Show sample data
            print(f"Sample data from Census lookup table:")
            sample_data = df[[lsoa_col, msoa_col]].head(5)
            print(sample_data)
            
            # Build the mapping
            mapping_count = 0
            for _, row in df.iterrows():
                lsoa_code = row[lsoa_col]
                msoa_code = row[msoa_col]
                
                if pd.notna(lsoa_code) and pd.notna(msoa_code):
                    lsoa_code = str(lsoa_code).strip()
                    msoa_code = str(msoa_code).strip()
                    
                    # Skip empty strings
                    if lsoa_code and msoa_code:
                        self.lsoa_msoa_mapping[lsoa_code] = msoa_code
                        
                        # Build reverse mapping
                        if msoa_code not in self.msoa_lsoa_mapping:
                            self.msoa_lsoa_mapping[msoa_code] = []
                        self.msoa_lsoa_mapping[msoa_code].append(lsoa_code)
                        mapping_count += 1
            
            print(f"✅ Processed {mapping_count} LSOA-MSOA mappings from Census lookup table")
            return len(self.lsoa_msoa_mapping) > 0
            
        except Exception as e:
            print(f"❌ Error loading from Census lookup table: {e}")
            return False
    
    def _load_fallback_mapping(self):
        """Load fallback mapping for common areas"""
        # This is a simplified fallback mapping for demonstration
        # In practice, you would want a more comprehensive mapping
        
        # Common London areas (simplified mapping)
        fallback_mapping = {
            # Westminster area LSOAs -> MSOAs
            'E01004730': 'E02000977',  # Westminster 001 -> Westminster 001
            'E01004731': 'E02000977',  # Westminster 002 -> Westminster 001
            'E01004732': 'E02000977',  # Westminster 003 -> Westminster 001
            'E01004733': 'E02000978',  # Westminster 004 -> Westminster 002
            'E01004734': 'E02000978',  # Westminster 005 -> Westminster 002
            
            # Kensington and Chelsea
            'E01004735': 'E02000979',  # Kensington 001 -> Kensington 001
            'E01004736': 'E02000979',  # Kensington 002 -> Kensington 001
            'E01004737': 'E02000980',  # Kensington 003 -> Kensington 002
            
            # Add more mappings as needed
        }
        
        self.lsoa_msoa_mapping.update(fallback_mapping)
        
        # Build reverse mapping
        for lsoa, msoa in fallback_mapping.items():
            if msoa not in self.msoa_lsoa_mapping:
                self.msoa_lsoa_mapping[msoa] = []
            self.msoa_lsoa_mapping[msoa].append(lsoa)
    
    def lsoa_to_msoa(self, lsoa_code: str) -> Optional[str]:
        """Convert LSOA code to MSOA code"""
        if not self.lsoa_msoa_mapping:
            self.load_mapping_data()
        
        return self.lsoa_msoa_mapping.get(lsoa_code)
    
    def msoa_to_lsoas(self, msoa_code: str) -> List[str]:
        """Get all LSOA codes for a given MSOA"""
        if not self.msoa_lsoa_mapping:
            self.load_mapping_data()
        
        return self.msoa_lsoa_mapping.get(msoa_code, [])
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about the mapping"""
        return {
            'total_lsoas': len(self.lsoa_msoa_mapping),
            'total_msoas': len(self.msoa_lsoa_mapping),
            'avg_lsoas_per_msoa': sum(len(lsoas) for lsoas in self.msoa_lsoa_mapping.values()) / len(self.msoa_lsoa_mapping) if self.msoa_lsoa_mapping else 0
        }
    
    def validate_mapping(self, lsoa_code: str, msoa_code: str) -> bool:
        """Validate that an LSOA belongs to an MSOA"""
        mapped_msoa = self.lsoa_to_msoa(lsoa_code)
        return mapped_msoa == msoa_code
    
    def export_mapping(self, format: str = 'json') -> str:
        """Export the mapping data"""
        if format == 'json':
            return json.dumps({
                'lsoa_to_msoa': self.lsoa_msoa_mapping,
                'msoa_to_lsoa': self.msoa_lsoa_mapping,
                'stats': self.get_mapping_stats()
            }, indent=2)
        elif format == 'csv':
            # Create a CSV with LSOA, MSOA pairs
            rows = []
            for lsoa, msoa in self.lsoa_msoa_mapping.items():
                rows.append(f"{lsoa},{msoa}")
            return "LSOA_CODE,MSOA_CODE\n" + "\n".join(rows)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global instance for easy access
lsoa_msoa_mapper = LSOAMSOAMapper()
