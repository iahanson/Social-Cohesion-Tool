#!/usr/bin/env python3
"""
Unemployment Data Connector

Handles loading and processing of unemployment data from the UK dataset.
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Any

class UnemploymentConnector:
    """Connector for unemployment data"""
    
    def __init__(self, data_file: str = "data/unmenploymentSept25.xls"):
        self.data_file = data_file
        self.unemployment_data = None
        self._data_loaded = False
        
        # Check if file exists
        if not os.path.exists(self.data_file):
            print(f"⚠️ Warning: Unemployment data file not found: {self.data_file}")
            print("   Available files in data directory:")
            try:
                data_files = [f for f in os.listdir("data") if f.endswith(('.xls', '.xlsx'))]
                for file in data_files:
                    print(f"   - {file}")
            except:
                print("   Could not list data directory")
    
    def load_data(self) -> bool:
        """Load unemployment data from Excel file"""
        try:
            if not os.path.exists(self.data_file):
                print(f"❌ Unemployment data file not found: {self.data_file}")
                return False
            
            print(f"📊 Loading unemployment data from {self.data_file}")
            
            # Read the Excel file
            print(f"📊 Reading Excel file: {self.data_file}")
            df = pd.read_excel(self.data_file)
            print(f"📊 Excel file read successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Clean the data - remove any header rows
            print(f"📊 Available columns: {list(df.columns)}")
            
            # Find geography code column - look for exact match first, then partial
            geo_code_col = None
            geo_name_col = None
            
            # Look for exact 'Geography code' match first
            for col in df.columns:
                if str(col).strip().lower() == 'geography code':
                    geo_code_col = col
                    break
            
            # If not found, look for partial matches
            if geo_code_col is None:
                for col in df.columns:
                    if 'code' in col.lower() and 'geography' in col.lower():
                        geo_code_col = col
                        break
            
            # Find geography name column (should be Column A or similar)
            for col in df.columns:
                if 'geography' in col.lower() and 'code' not in col.lower():
                    geo_name_col = col
                    break
            
            # If still not found, try common patterns
            if geo_name_col is None:
                # Try first column as geography name
                if len(df.columns) > 0:
                    geo_name_col = df.columns[0]
            
            print(f"📊 Found geography code column: {geo_code_col}")
            print(f"📊 Found geography name column: {geo_name_col}")
            
            if geo_code_col is None or geo_name_col is None:
                print(f"❌ Could not find geography columns. Available: {list(df.columns)}")
                return False
            
            # Find the first row with actual data (not headers)
            data_start_row = 0
            for idx, row in df.iterrows():
                if pd.notna(row[geo_code_col]) and str(row[geo_code_col]).strip():
                    data_start_row = idx
                    break
            
            # Extract data starting from the first data row
            self.unemployment_data = df.iloc[data_start_row:].copy()
            
            # Clean column names
            self.unemployment_data.columns = [str(col).strip() for col in self.unemployment_data.columns]
            
            # Rename key columns for consistency
            column_mapping = {
                geo_code_col: 'geography_code',
                geo_name_col: 'geography_name'
            }
            
            # Add other important columns (E and H as mentioned)
            # Column E (index 4) - people looking for work
            if len(self.unemployment_data.columns) > 4:
                col_e = self.unemployment_data.columns[4]
                column_mapping[col_e] = 'people_looking_for_work'
                print(f"📊 Column E ({col_e}) mapped to 'people_looking_for_work'")
            
            # Column H (index 7) - unemployment proportion
            if len(self.unemployment_data.columns) > 7:
                col_h = self.unemployment_data.columns[7]
                column_mapping[col_h] = 'unemployment_proportion'
                print(f"📊 Column H ({col_h}) mapped to 'unemployment_proportion'")
            
            self.unemployment_data = self.unemployment_data.rename(columns=column_mapping)
            
            # Clean the data
            self.unemployment_data = self.unemployment_data.dropna(subset=['geography_code', 'geography_name'])
            
            # Convert numeric columns
            numeric_cols = ['people_looking_for_work', 'unemployment_proportion']
            for col in numeric_cols:
                if col in self.unemployment_data.columns:
                    self.unemployment_data[col] = pd.to_numeric(self.unemployment_data[col], errors='coerce')
            
            self._data_loaded = True
            print(f"✅ Loaded unemployment data: {len(self.unemployment_data)} areas")
            print(f"📋 Columns: {list(self.unemployment_data.columns)}")
            
            # Show sample data
            if len(self.unemployment_data) > 0:
                print("📊 Sample unemployment data:")
                print(self.unemployment_data.head(3))
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading unemployment data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_unemployment_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded unemployment data"""
        if not self._data_loaded:
            self.load_data()
        return self.unemployment_data
    
    def get_unemployment_by_lad(self, lad_name: str) -> Optional[Dict[str, Any]]:
        """Get unemployment data for a specific LAD"""
        if not self._data_loaded:
            self.load_data()
        
        if self.unemployment_data is None:
            return None
        
        # Try exact match first
        exact_match = self.unemployment_data[
            self.unemployment_data['geography_name'].str.lower() == lad_name.lower()
        ]
        
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return {
                'geography_code': row.get('geography_code'),
                'geography_name': row.get('geography_name'),
                'people_looking_for_work': row.get('people_looking_for_work', 0),
                'unemployment_proportion': row.get('unemployment_proportion', 0),
                'match_type': 'exact'
            }
        
        # Try partial match
        partial_match = self.unemployment_data[
            self.unemployment_data['geography_name'].str.contains(lad_name, case=False, na=False)
        ]
        
        if not partial_match.empty:
            row = partial_match.iloc[0]
            return {
                'geography_code': row.get('geography_code'),
                'geography_name': row.get('geography_name'),
                'people_looking_for_work': row.get('people_looking_for_work', 0),
                'unemployment_proportion': row.get('unemployment_proportion', 0),
                'match_type': 'partial'
            }
        
        return None
    
    def get_unemployment_summary(self) -> Dict[str, Any]:
        """Get summary statistics for unemployment data"""
        if not self._data_loaded:
            self.load_data()
        
        if self.unemployment_data is None:
            return {}
        
        summary = {
            'total_areas': len(self.unemployment_data),
            'total_people_looking_for_work': self.unemployment_data['people_looking_for_work'].sum() if 'people_looking_for_work' in self.unemployment_data.columns else 0,
            'average_unemployment_proportion': self.unemployment_data['unemployment_proportion'].mean() if 'unemployment_proportion' in self.unemployment_data.columns else 0,
            'max_unemployment_proportion': self.unemployment_data['unemployment_proportion'].max() if 'unemployment_proportion' in self.unemployment_data.columns else 0,
            'min_unemployment_proportion': self.unemployment_data['unemployment_proportion'].min() if 'unemployment_proportion' in self.unemployment_data.columns else 0
        }
        
        return summary
    
    def get_all_lads(self) -> List[str]:
        """Get list of all LAD names in the unemployment data"""
        if not self._data_loaded:
            self.load_data()
        
        if self.unemployment_data is None:
            return []
        
        return sorted(self.unemployment_data['geography_name'].unique().tolist())
