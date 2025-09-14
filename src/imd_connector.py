"""
Indices of Multiple Deprivation (IMD) Data Connector
Simplified connector for real IMD data processing
"""

import pandas as pd
import os
from typing import Optional, Dict, Any, List

class IMDConnector:
    """Handles IMD data processing from local file"""
    
    def __init__(self):
        # Set path to the IMD data file
        self.data_dir = "data"
        self.imd_file = os.path.join(self.data_dir, "IMD2019_Index_of_Multiple_Deprivation.xlsx")
        
        # Cache for loaded data
        self._data_cache = None
    
    def _check_data_file(self) -> bool:
        """
        Check if the IMD data file exists
        
        Returns:
            True if file exists, False otherwise
        """
        if not os.path.exists(self.imd_file):
            print(f"IMD data file not found at: {self.imd_file}")
            print("Please ensure the IMD2019_Index_of_Multiple_Deprivation.xlsx file is in the data/ folder")
            return False
        return True
    
    def load_imd_data(self) -> Optional[pd.DataFrame]:
        """
        Load IMD data from local file
        
        Returns:
            DataFrame containing IMD data or None if error
        """
        # Return cached data if available
        if self._data_cache is not None:
            return self._data_cache
        
        # Check if file exists
        if not self._check_data_file():
            return None
        
        try:
            print(f"Loading IMD data from {self.imd_file}")
            
            # Load the specific sheet 'IMD2019'
            df = pd.read_excel(self.imd_file, sheet_name='IMD2019')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Print column information for verification
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Verify we have the expected columns
            expected_columns = [
                'LSOA code (2011)',
                'LSOA name (2011)', 
                'Local Authority District code (2019)',
                'Local Authority District name (2019)',
                'Index of Multiple Deprivation (IMD) Rank',
                'Index of Multiple Deprivation (IMD) Decile'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing expected columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                return None
            
            # Cache the data
            self._data_cache = df
            print("IMD data loaded successfully")
            return df
            
        except Exception as e:
            print(f"Error loading IMD data: {e}")
            return None
    
    def get_imd_for_msoa(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get IMD data for a specific MSOA
        
        Args:
            msoa_code: MSOA code to look up
            
        Returns:
            Dictionary containing IMD data for the MSOA or None if not found
        """
        try:
            df = self.load_imd_data()
            if df is None:
                return None
            
            # Find LSOAs that belong to this MSOA (first 9 characters match)
            lsoa_codes_for_msoa = df[df['LSOA code (2011)'].str.startswith(msoa_code[:9])]['LSOA code (2011)'].tolist()
            
            if not lsoa_codes_for_msoa:
                print(f"No LSOAs found for MSOA {msoa_code}")
                return None
            
            # Filter data for LSOAs in this MSOA
            msoa_data = df[df['LSOA code (2011)'].isin(lsoa_codes_for_msoa)]
            
            if msoa_data.empty:
                return None
            
            # Calculate MSOA-level statistics
            imd_data = {
                'msoa_code': msoa_code,
                'total_lsoas': len(msoa_data),
                'lsoa_codes': lsoa_codes_for_msoa,
                'local_authority_code': msoa_data['Local Authority District code (2019)'].iloc[0],
                'local_authority_name': msoa_data['Local Authority District name (2019)'].iloc[0]
            }
            
            # Add IMD decile (primary metric)
            imd_data['imd_decile'] = msoa_data['Index of Multiple Deprivation (IMD) Decile'].mean()
            imd_data['imd_decile_min'] = msoa_data['Index of Multiple Deprivation (IMD) Decile'].min()
            imd_data['imd_decile_max'] = msoa_data['Index of Multiple Deprivation (IMD) Decile'].max()
            
            # Add IMD rank (auxiliary data)
            imd_data['imd_rank'] = msoa_data['Index of Multiple Deprivation (IMD) Rank'].mean()
            imd_data['imd_rank_min'] = msoa_data['Index of Multiple Deprivation (IMD) Rank'].min()
            imd_data['imd_rank_max'] = msoa_data['Index of Multiple Deprivation (IMD) Rank'].max()
            
            return imd_data
            
        except Exception as e:
            print(f"Error getting IMD data for MSOA {msoa_code}: {e}")
            return None
    
    def get_imd_for_lsoa(self, lsoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get IMD data for a specific LSOA
        
        Args:
            lsoa_code: LSOA code to look up
            
        Returns:
            Dictionary containing IMD data for the LSOA or None if not found
        """
        try:
            df = self.load_imd_data()
            if df is None:
                return None
            
            # Find the specific LSOA
            lsoa_data = df[df['LSOA code (2011)'] == lsoa_code]
            
            if lsoa_data.empty:
                return None
            
            # Get the row data
            row = lsoa_data.iloc[0]
            
            # Create IMD data dictionary
            imd_data = {
                'lsoa_code': lsoa_code,
                'lsoa_name': row['LSOA name (2011)'],
                'local_authority_code': row['Local Authority District code (2019)'],
                'local_authority_name': row['Local Authority District name (2019)'],
                'imd_decile': row['Index of Multiple Deprivation (IMD) Decile'],
                'imd_rank': row['Index of Multiple Deprivation (IMD) Rank']
            }
            
            return imd_data
            
        except Exception as e:
            print(f"Error getting IMD data for LSOA {lsoa_code}: {e}")
            return None
    
    def get_imd_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for IMD data
        
        Returns:
            Dictionary containing summary statistics
        """
        try:
            df = self.load_imd_data()
            if df is None:
                return None
            
            summary = {
                'total_lsoas': len(df),
                'date': '2019',
                'source': 'English Indices of Deprivation 2019',
                'data_file': self.imd_file
            }
            
            # IMD decile statistics (primary metric)
            summary['average_imd_decile'] = df['Index of Multiple Deprivation (IMD) Decile'].mean()
            summary['imd_decile_distribution'] = df['Index of Multiple Deprivation (IMD) Decile'].value_counts().sort_index().to_dict()
            
            # Find most and least deprived LSOAs by decile
            most_deprived_idx = df['Index of Multiple Deprivation (IMD) Decile'].idxmin()
            least_deprived_idx = df['Index of Multiple Deprivation (IMD) Decile'].idxmax()
            
            summary['most_deprived_lsoa'] = {
                'name': df.loc[most_deprived_idx, 'LSOA name (2011)'],
                'code': df.loc[most_deprived_idx, 'LSOA code (2011)'],
                'decile': df.loc[most_deprived_idx, 'Index of Multiple Deprivation (IMD) Decile'],
                'local_authority': df.loc[most_deprived_idx, 'Local Authority District name (2019)']
            }
            
            summary['least_deprived_lsoa'] = {
                'name': df.loc[least_deprived_idx, 'LSOA name (2011)'],
                'code': df.loc[least_deprived_idx, 'LSOA code (2011)'],
                'decile': df.loc[least_deprived_idx, 'Index of Multiple Deprivation (IMD) Decile'],
                'local_authority': df.loc[least_deprived_idx, 'Local Authority District name (2019)']
            }
            
            # IMD rank statistics (auxiliary data)
            summary['average_imd_rank'] = df['Index of Multiple Deprivation (IMD) Rank'].mean()
            summary['imd_rank_range'] = {
                'min': df['Index of Multiple Deprivation (IMD) Rank'].min(),
                'max': df['Index of Multiple Deprivation (IMD) Rank'].max()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting IMD summary: {e}")
            return None
    
    def get_imd_for_multiple_msoas(self, msoa_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get IMD data for multiple MSOAs
        
        Args:
            msoa_codes: List of MSOA codes to look up
            
        Returns:
            Dictionary mapping MSOA codes to their IMD data
        """
        results = {}
        for msoa_code in msoa_codes:
            imd_data = self.get_imd_for_msoa(msoa_code)
            if imd_data:
                results[msoa_code] = imd_data
        return results
    
    def get_imd_for_multiple_lsoas(self, lsoa_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get IMD data for multiple LSOAs
        
        Args:
            lsoa_codes: List of LSOA codes to look up
            
        Returns:
            Dictionary mapping LSOA codes to their IMD data
        """
        results = {}
        for lsoa_code in lsoa_codes:
            imd_data = self.get_imd_for_lsoa(lsoa_code)
            if imd_data:
                results[lsoa_code] = imd_data
        return results
