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
        self.imd_file = get_local_file_path('imd_data') or os.path.join(self.data_dir, "IMD2019_Index_of_Multiple_Deprivation.xlsx")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Cache for LSOA to MSOA mapping
        self._lsoa_to_msoa_cache = None
    
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
            
            # Load the Excel file - try different sheet names
            print(f"Loading IMD data from {self.imd_file}")
            
            # Try to read the file and detect the correct sheet
            excel_file = pd.ExcelFile(self.imd_file)
            print(f"Available sheets: {excel_file.sheet_names}")
            
            # Try common sheet names
            sheet_names_to_try = ['IMD2019', 'IMD 2019', 'Sheet1', 'Data', 'Index of Multiple Deprivation']
            df = None
            
            for sheet_name in sheet_names_to_try:
                if sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(self.imd_file, sheet_name=sheet_name)
                        print(f"Successfully loaded data from sheet: {sheet_name}")
                        break
                    except Exception as e:
                        print(f"Failed to load sheet {sheet_name}: {e}")
                        continue
            
            if df is None:
                # Try the first sheet
                df = pd.read_excel(self.imd_file, sheet_name=0)
                print("Loaded data from first sheet")
            
            # Clean column names and standardize
            df.columns = df.columns.str.strip()
            
            # Print column information for debugging
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Ensure we have the required columns
            required_columns = ['LSOA code (2011)', 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                # Try to find similar column names
                for missing_col in missing_columns:
                    similar_cols = [col for col in df.columns if missing_col.lower().replace(' ', '') in col.lower().replace(' ', '')]
                    if similar_cols:
                        print(f"Similar columns found for '{missing_col}': {similar_cols}")
            
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
    
    def _create_lsoa_to_msoa_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Create mapping from LSOA codes to MSOA codes
        
        Args:
            df: DataFrame containing LSOA data
            
        Returns:
            Dictionary mapping LSOA codes to MSOA codes
        """
        if self._lsoa_to_msoa_cache is not None:
            return self._lsoa_to_msoa_cache
        
        mapping = {}
        
        # LSOA codes are 9 characters, MSOA codes are also 9 characters
        # The first 9 characters of an LSOA code should correspond to an MSOA code
        # This is a simplified mapping - in reality, you might need a proper lookup table
        
        for lsoa_code in df['LSOA code (2011)'].unique():
            if pd.isna(lsoa_code):
                continue
            
            # Convert LSOA to MSOA by taking the first 9 characters
            # This is a simplified approach - you might need a proper mapping file
            msoa_code = lsoa_code[:9]
            mapping[lsoa_code] = msoa_code
        
        self._lsoa_to_msoa_cache = mapping
        return mapping
    
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
            
            # Create LSOA to MSOA mapping
            lsoa_to_msoa = self._create_lsoa_to_msoa_mapping(df)
            
            # Find LSOAs that belong to this MSOA
            # Method 1: Use the mapping if available
            lsoa_codes_for_msoa = [lsoa for lsoa, msoa in lsoa_to_msoa.items() if msoa == msoa_code]
            
            # Method 2: Fallback to string matching (first 9 characters)
            if not lsoa_codes_for_msoa:
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
                'lsoa_codes': lsoa_codes_for_msoa
            }
            
            # Add IMD decile (primary metric you mentioned)
            if 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)' in msoa_data.columns:
                imd_data['imd_decile'] = msoa_data['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'].mean()
                imd_data['imd_decile_min'] = msoa_data['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'].min()
                imd_data['imd_decile_max'] = msoa_data['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'].max()
            
            # Add IMD rank if available
            if 'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)' in msoa_data.columns:
                imd_data['imd_rank'] = msoa_data['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'].mean()
            
            # Add other domain ranks if available
            domain_columns = {
                'income_rank': 'Income Rank (where 1 is most deprived)',
                'employment_rank': 'Employment Rank (where 1 is most deprived)',
                'education_rank': 'Education, Skills and Training Rank (where 1 is most deprived)',
                'health_rank': 'Health Deprivation and Disability Rank (where 1 is most deprived)',
                'crime_rank': 'Crime Rank (where 1 is most deprived)',
                'barriers_rank': 'Barriers to Housing and Services Rank (where 1 is most deprived)',
                'living_rank': 'Living Environment Rank (where 1 is most deprived)'
            }
            
            for key, column_name in domain_columns.items():
                if column_name in msoa_data.columns:
                    imd_data[key] = msoa_data[column_name].mean()
            
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
                'lsoa_name': row.get('LSOA name (2011)', 'Unknown')
            }
            
            # Add IMD decile (primary metric)
            if 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)' in row:
                imd_data['imd_decile'] = row['Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)']
            
            # Add IMD rank if available
            if 'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)' in row:
                imd_data['imd_rank'] = row['Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)']
            
            # Add other domain ranks if available
            domain_columns = {
                'income_rank': 'Income Rank (where 1 is most deprived)',
                'employment_rank': 'Employment Rank (where 1 is most deprived)',
                'education_rank': 'Education, Skills and Training Rank (where 1 is most deprived)',
                'health_rank': 'Health Deprivation and Disability Rank (where 1 is most deprived)',
                'crime_rank': 'Crime Rank (where 1 is most deprived)',
                'barriers_rank': 'Barriers to Housing and Services Rank (where 1 is most deprived)',
                'living_rank': 'Living Environment Rank (where 1 is most deprived)'
            }
            
            for key, column_name in domain_columns.items():
                if column_name in row:
                    imd_data[key] = row[column_name]
            
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
            
            # Add IMD decile statistics (primary metric)
            if 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)' in df.columns:
                decile_col = 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'
                summary['average_imd_decile'] = df[decile_col].mean()
                summary['imd_decile_distribution'] = df[decile_col].value_counts().sort_index().to_dict()
                
                # Find most and least deprived LSOAs by decile
                most_deprived_idx = df[decile_col].idxmin()
                least_deprived_idx = df[decile_col].idxmax()
                
                summary['most_deprived_lsoa'] = {
                    'name': df.loc[most_deprived_idx, 'LSOA name (2011)'] if 'LSOA name (2011)' in df.columns else 'Unknown',
                    'code': df.loc[most_deprived_idx, 'LSOA code (2011)'],
                    'decile': df.loc[most_deprived_idx, decile_col]
                }
                
                summary['least_deprived_lsoa'] = {
                    'name': df.loc[least_deprived_idx, 'LSOA name (2011)'] if 'LSOA name (2011)' in df.columns else 'Unknown',
                    'code': df.loc[least_deprived_idx, 'LSOA code (2011)'],
                    'decile': df.loc[least_deprived_idx, decile_col]
                }
            
            # Add IMD rank statistics if available
            if 'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)' in df.columns:
                rank_col = 'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)'
                summary['average_imd_rank'] = df[rank_col].mean()
                summary['imd_rank_range'] = {
                    'min': df[rank_col].min(),
                    'max': df[rank_col].max()
                }
            
            # Add domain statistics if available
            domain_columns = {
                'income': 'Income Rank (where 1 is most deprived)',
                'employment': 'Employment Rank (where 1 is most deprived)',
                'education': 'Education, Skills and Training Rank (where 1 is most deprived)',
                'health': 'Health Deprivation and Disability Rank (where 1 is most deprived)',
                'crime': 'Crime Rank (where 1 is most deprived)',
                'barriers': 'Barriers to Housing and Services Rank (where 1 is most deprived)',
                'living': 'Living Environment Rank (where 1 is most deprived)'
            }
            
            domain_stats = {}
            for domain, column_name in domain_columns.items():
                if column_name in df.columns:
                    domain_stats[domain] = {
                        'average_rank': df[column_name].mean(),
                        'min_rank': df[column_name].min(),
                        'max_rank': df[column_name].max()
                    }
            
            if domain_stats:
                summary['domain_statistics'] = domain_stats
            
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
