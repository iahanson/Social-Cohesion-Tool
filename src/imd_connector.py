"""
Indices of Multiple Deprivation (IMD) Data Connector
Handles downloading and processing IMD data for MSOAs
"""

import requests
import pandas as pd
import os
from typing import Optional, Dict, Any, List
import zipfile
import io
from .data_config import get_data_config, use_real_data, use_dummy_data, get_data_url, get_local_file_path

class IMDConnector:
    """Handles IMD data retrieval and processing"""
    
    def __init__(self):
        # Get configuration from data config system
        self.data_config = get_data_config()
        self.config = self.data_config.get_config('imd_data')
        
        # Set URLs and paths from configuration
        self.imd_data_url = get_data_url('imd_data') or "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833978/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx"
        self.imd_metadata_url = "https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019"
        self.data_dir = "data"
        self.imd_file = get_local_file_path('imd_data') or os.path.join(self.data_dir, "imd_2019.xlsx")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_imd_data(self) -> bool:
        """
        Download IMD 2019 data from GOV.UK
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Downloading IMD 2019 data...")
            response = requests.get(self.imd_data_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(self.imd_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"IMD data downloaded to {self.imd_file}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("IMD data URL not found. This is expected - using sample data instead.")
                print("For real IMD data, please download manually from:")
                print("https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019")
                return False
            else:
                print(f"HTTP error downloading IMD data: {e}")
                return False
        except Exception as e:
            print(f"Error downloading IMD data: {e}")
            print("Using sample data instead.")
            return False
    
    def load_imd_data(self) -> Optional[pd.DataFrame]:
        """
        Load IMD data from local file
        
        Returns:
            DataFrame containing IMD data or None if error
        """
        # Check configuration to determine data source
        if use_dummy_data('imd_data'):
            print("Using sample IMD data (configured for dummy data)")
            return self._generate_sample_imd_data()
        
        try:
            if not os.path.exists(self.imd_file):
                print("IMD data not found locally. Downloading...")
                if not self.download_imd_data():
                    if self.config.fallback_to_dummy:
                        print("Download failed. Using sample IMD data for demonstration...")
                        return self._generate_sample_imd_data()
                    else:
                        raise Exception("IMD data download failed and fallback disabled")
            
            # Load the Excel file
            print(f"Loading IMD data from {self.imd_file}")
            df = pd.read_excel(self.imd_file, sheet_name='IMD2019')
            return df
            
        except Exception as e:
            print(f"Error loading IMD data: {e}")
            if self.config.fallback_to_dummy:
                print("Using sample IMD data for demonstration...")
                return self._generate_sample_imd_data()
            else:
                raise Exception(f"IMD data loading failed: {e}")
    
    def _generate_sample_imd_data(self) -> pd.DataFrame:
        """
        Generate sample IMD data for demonstration purposes
        
        Returns:
            DataFrame with sample IMD data
        """
        import numpy as np
        
        # Generate sample LSOA data (IMD is calculated at LSOA level)
        n_lsoas = 1000
        data = {
            'LSOA code (2011)': [f'E0100{i:05d}' for i in range(1, n_lsoas + 1)],
            'LSOA name (2011)': [f'Sample LSOA {i:05d}' for i in range(1, n_lsoas + 1)],
            'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': np.random.randint(1, 11, n_lsoas),
            'Income Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Employment Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Education, Skills and Training Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Health Deprivation and Disability Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Crime Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Barriers to Housing and Services Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas),
            'Living Environment Rank (where 1 is most deprived)': np.random.randint(1, 32845, n_lsoas)
        }
        
        return pd.DataFrame(data)
    
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
            
            # Filter for the specific MSOA
            msoa_data = df[df['LSOA code (2011)'].str.startswith(msoa_code[:9])]
            
            if msoa_data.empty:
                return None
            
            # Calculate MSOA-level statistics (average across LSOAs)
            imd_data = {
                'msoa_code': msoa_code,
                'total_lsoas': len(msoa_data),
                'imd_rank': msoa_data['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'].mean(),
                'imd_decile': msoa_data['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'].mean(),
                'income_rank': msoa_data['Income Rank (where 1 is most deprived)'].mean(),
                'employment_rank': msoa_data['Employment Rank (where 1 is most deprived)'].mean(),
                'education_rank': msoa_data['Education, Skills and Training Rank (where 1 is most deprived)'].mean(),
                'health_rank': msoa_data['Health Deprivation and Disability Rank (where 1 is most deprived)'].mean(),
                'crime_rank': msoa_data['Crime Rank (where 1 is most deprived)'].mean(),
                'barriers_rank': msoa_data['Barriers to Housing and Services Rank (where 1 is most deprived)'].mean(),
                'living_rank': msoa_data['Living Environment Rank (where 1 is most deprived)'].mean(),
                'lsoa_codes': msoa_data['LSOA code (2011)'].tolist()
            }
            
            return imd_data
            
        except Exception as e:
            print(f"Error getting IMD data for MSOA {msoa_code}: {e}")
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
                'average_imd_rank': df['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'].mean(),
                'most_deprived_lsoa': df.loc[df['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'].idxmin(), 'LSOA name (2011)'],
                'least_deprived_lsoa': df.loc[df['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'].idxmax(), 'LSOA name (2011)']
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting IMD summary: {e}")
            return None
