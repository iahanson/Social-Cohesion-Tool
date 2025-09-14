"""
Good Neighbours Social Trust Data Connector
Handles loading and processing of social trust data by MSOA
"""

import pandas as pd
import os
from typing import Optional, Dict, Any, List

class GoodNeighboursConnector:
    """Handles Good Neighbours social trust data processing from local file"""
    
    def __init__(self):
        # Set path to the Good Neighbours data file
        self.data_dir = "data"
        self.data_file = os.path.join(self.data_dir, "good_neighbours_full_data_by_msoa.xlsx")
        
        # Cache for loaded data
        self._data_cache = None
    
    def _check_data_file(self) -> bool:
        """
        Check if the Good Neighbours data file exists
        
        Returns:
            True if file exists, False otherwise
        """
        if not os.path.exists(self.data_file):
            print(f"Good Neighbours data file not found at: {self.data_file}")
            print("Please ensure the good_neighbours_full_data_by_msoa.xlsx file is in the data/ folder")
            return False
        return True
    
    def load_social_trust_data(self) -> Optional[pd.DataFrame]:
        """
        Load Good Neighbours social trust data from local file
        
        Returns:
            DataFrame containing social trust data or None if error
        """
        # Return cached data if available
        if self._data_cache is not None:
            return self._data_cache
        
        # Check if file exists
        if not self._check_data_file():
            return None
        
        try:
            print(f"Loading Good Neighbours social trust data from {self.data_file}")
            
            # Load the Excel file (assuming first sheet contains the data)
            df = pd.read_excel(self.data_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Print column information for verification
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Verify we have the expected columns
            expected_columns = [
                'MSOA_code',
                'MSOA_name',
                'always_trust OR usually_trust',
                'usually_careful OR almost_always_careful',
                'Net_trust'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing expected columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                return None
            
            # Cache the data
            self._data_cache = df
            print("Good Neighbours social trust data loaded successfully")
            return df
            
        except Exception as e:
            print(f"Error loading Good Neighbours social trust data: {e}")
            return None
    
    def get_social_trust_for_msoa(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get social trust data for a specific MSOA
        
        Args:
            msoa_code: MSOA code to look up
            
        Returns:
            Dictionary containing social trust data for the MSOA or None if not found
        """
        try:
            df = self.load_social_trust_data()
            if df is None:
                return None
            
            # Find the specific MSOA
            msoa_data = df[df['MSOA_code'] == msoa_code]
            
            if msoa_data.empty:
                print(f"No social trust data found for MSOA {msoa_code}")
                return None
            
            # Get the row data
            row = msoa_data.iloc[0]
            
            # Create social trust data dictionary
            trust_data = {
                'msoa_code': msoa_code,
                'msoa_name': row['MSOA_name'],
                'always_usually_trust': row['always_trust OR usually_trust'],
                'usually_almost_always_careful': row['usually_careful OR almost_always_careful'],
                'net_trust': row['Net_trust']
            }
            
            return trust_data
            
        except Exception as e:
            print(f"Error getting social trust data for MSOA {msoa_code}: {e}")
            return None
    
    def get_social_trust_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for social trust data
        
        Returns:
            Dictionary containing summary statistics
        """
        try:
            df = self.load_social_trust_data()
            if df is None:
                return None
            
            summary = {
                'total_msoas': len(df),
                'source': 'Good Neighbours Social Trust Survey',
                'data_file': self.data_file
            }
            
            # Net trust statistics
            summary['average_net_trust'] = df['Net_trust'].mean()
            summary['net_trust_range'] = {
                'min': df['Net_trust'].min(),
                'max': df['Net_trust'].max(),
                'std': df['Net_trust'].std()
            }
            
            # Trust distribution
            summary['net_trust_distribution'] = {
                'positive_trust': len(df[df['Net_trust'] > 0]),
                'negative_trust': len(df[df['Net_trust'] < 0]),
                'neutral_trust': len(df[df['Net_trust'] == 0])
            }
            
            # Find highest and lowest trust MSOAs
            highest_trust_idx = df['Net_trust'].idxmax()
            lowest_trust_idx = df['Net_trust'].idxmin()
            
            summary['highest_trust_msoa'] = {
                'name': df.loc[highest_trust_idx, 'MSOA_name'],
                'code': df.loc[highest_trust_idx, 'MSOA_code'],
                'net_trust': df.loc[highest_trust_idx, 'Net_trust']
            }
            
            summary['lowest_trust_msoa'] = {
                'name': df.loc[lowest_trust_idx, 'MSOA_name'],
                'code': df.loc[lowest_trust_idx, 'MSOA_code'],
                'net_trust': df.loc[lowest_trust_idx, 'Net_trust']
            }
            
            # Trust component statistics
            summary['trust_components'] = {
                'average_always_usually_trust': df['always_trust OR usually_trust'].mean(),
                'average_usually_almost_always_careful': df['usually_careful OR almost_always_careful'].mean()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting social trust summary: {e}")
            return None
    
    def get_social_trust_for_multiple_msoas(self, msoa_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get social trust data for multiple MSOAs
        
        Args:
            msoa_codes: List of MSOA codes to look up
            
        Returns:
            Dictionary mapping MSOA codes to their social trust data
        """
        results = {}
        for msoa_code in msoa_codes:
            trust_data = self.get_social_trust_for_msoa(msoa_code)
            if trust_data:
                results[msoa_code] = trust_data
        return results
    
    def get_top_trust_msoas(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N MSOAs by net trust score
        
        Args:
            n: Number of top MSOAs to return
            
        Returns:
            List of dictionaries containing top trust MSOAs
        """
        try:
            df = self.load_social_trust_data()
            if df is None:
                return []
            
            # Get top N MSOAs by net trust
            top_msoas = df.nlargest(n, 'Net_trust')
            
            results = []
            for idx, row in top_msoas.iterrows():
                results.append({
                    'msoa_code': row['MSOA_code'],
                    'msoa_name': row['MSOA_name'],
                    'net_trust': row['Net_trust'],
                    'always_usually_trust': row['always_trust OR usually_trust'],
                    'usually_almost_always_careful': row['usually_careful OR almost_always_careful']
                })
            
            return results
            
        except Exception as e:
            print(f"Error getting top trust MSOAs: {e}")
            return []
    
    def get_lowest_trust_msoas(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get bottom N MSOAs by net trust score
        
        Args:
            n: Number of bottom MSOAs to return
            
        Returns:
            List of dictionaries containing lowest trust MSOAs
        """
        try:
            df = self.load_social_trust_data()
            if df is None:
                return []
            
            # Get bottom N MSOAs by net trust
            bottom_msoas = df.nsmallest(n, 'Net_trust')
            
            results = []
            for idx, row in bottom_msoas.iterrows():
                results.append({
                    'msoa_code': row['MSOA_code'],
                    'msoa_name': row['MSOA_name'],
                    'net_trust': row['Net_trust'],
                    'always_usually_trust': row['always_trust OR usually_trust'],
                    'usually_almost_always_careful': row['usually_careful OR almost_always_careful']
                })
            
            return results
            
        except Exception as e:
            print(f"Error getting lowest trust MSOAs: {e}")
            return []
