"""
Unified Data Connector
Combines functionality from IMD and Good Neighbours connectors
Provides a single interface for all MSOA-based data sources
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
from .lsoa_msoa_mapper import lsoa_msoa_mapper
from .community_life_survey_connector import CommunityLifeSurveyConnector
from .unemployment_connector import UnemploymentConnector

@dataclass
class MSOADataResult:
    """Result container for MSOA data queries"""
    msoa_code: str
    msoa_name: str
    data_source: str
    data: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class UnifiedDataConnector:
    """Unified connector for all MSOA-based data sources"""
    
    def __init__(self, auto_load=True):
        self.data_config = self._load_data_config()
        self.imd_data = None
        self.good_neighbours_data = None
        self.population_data = None
        self.msoa_population_data = None
        self.demographic_columns = []
        self.population_cache_file = os.path.join("data", "msoa_population_cache.json")
        self.community_survey_data = None
        self.community_survey_connector = CommunityLifeSurveyConnector()
        self.unemployment_data = None
        try:
            self.unemployment_connector = UnemploymentConnector()
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize unemployment connector: {e}")
            self.unemployment_connector = None
        
        if auto_load:
            self._load_data_sources()
    
    def _load_data_config(self) -> Dict[str, Any]:
        """Load data configuration"""
        return {
            'imd': {
                'enabled': os.getenv('IMD_DATA_USE_REAL_DATA', 'true').lower() == 'true',
                'file_path': os.getenv('IMD_DATA_FILE_PATH', 'data/IMD2019_Index_of_Multiple_Deprivation.xlsx'),
                'sheet_name': 'IMD2019'
            },
            'good_neighbours': {
                'enabled': os.getenv('GOOD_NEIGHBOURS_USE_REAL_DATA', 'true').lower() == 'true',
                'file_path': os.getenv('GOOD_NEIGHBOURS_FILE_PATH', 'data/good_neighbours_full_data_by_msoa.xlsx'),
                'sheet_name': 0  # First sheet
            },
            'population': {
                'enabled': os.getenv('POPULATION_DATA_ENABLED', 'true').lower() == 'true',
                'file_path': os.getenv('POPULATION_DATA_FILE_PATH', 'data/Census_population_2022.xlsx'),
                'sheet_name': 'Mid-2022 LSOA 2021'
            },
            'community_survey': {
                'enabled': os.getenv('COMMUNITY_SURVEY_ENABLED', 'true').lower() == 'true',
                'file_path': os.getenv('COMMUNITY_SURVEY_FILE_PATH', 'data/Community_Life_Survey_2023_24.xlsx')
            }
        }
    
    def _load_data_sources(self):
        """Load all available data sources"""
        try:
            if self.data_config['imd']['enabled']:
                self.imd_data = self._load_imd_data()
                print("✅ IMD data loaded successfully")
            else:
                print("⚠️ IMD data disabled")
            
            if self.data_config['good_neighbours']['enabled']:
                self.good_neighbours_data = self._load_good_neighbours_data()
                print("✅ Good Neighbours data loaded successfully")
            else:
                print("⚠️ Good Neighbours data disabled")
            
            if self.data_config['population']['enabled']:
                # Try to load from cache first
                if self._load_population_from_cache():
                    print("✅ Population data loaded from cache")
                else:
                    # Load from original file and create cache
                    self.population_data = self._load_population_data()
                    if self.population_data is not None:
                        print("✅ Population data loaded successfully")
                        self._aggregate_population_to_msoa()
                        self._save_population_cache()
                    else:
                        print("❌ Failed to load Population data")
            else:
                print("⚠️ Population data disabled")
            
            if self.data_config['community_survey']['enabled']:
                self.community_survey_data = self._load_community_survey_data()
                if self.community_survey_data is not None:
                    print("✅ Community Life Survey data loaded successfully")
                else:
                    print("❌ Community Life Survey data loading failed")
            else:
                print("⚠️ Community Life Survey data disabled")
            
            # Load unemployment data
            self.unemployment_data = self._load_unemployment_data()
            if self.unemployment_data is not None:
                print("✅ Unemployment data loaded successfully")
            else:
                print("❌ Unemployment data loading failed")
                
        except Exception as e:
            print(f"❌ Error loading data sources: {e}")
    
    def _load_imd_data(self) -> Optional[pd.DataFrame]:
        """Load IMD data from Excel file and aggregate to MSOA level"""
        try:
            file_path = self.data_config['imd']['file_path']
            if not os.path.exists(file_path):
                print(f"❌ IMD file not found: {file_path}")
                return None
            
            df = pd.read_excel(file_path, sheet_name=self.data_config['imd']['sheet_name'])
            
            # Standardize column names
            column_mapping = {
                'LSOA code (2011)': 'lsoa_code',
                'LSOA name (2011)': 'lsoa_name',
                'Local Authority District code (2019)': 'la_code',
                'Local Authority District name (2019)': 'la_name',
                'Index of Multiple Deprivation (IMD) Rank': 'imd_rank',
                'Index of Multiple Deprivation (IMD) Decile': 'imd_decile'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Load LSOA to MSOA mapping
            if not lsoa_msoa_mapper.load_mapping_data():
                print("⚠️ Could not load LSOA-MSOA mapping, using fallback")
            
            # Map LSOA codes to MSOA codes
            df['msoa_code'] = df['lsoa_code'].apply(
                lambda lsoa: lsoa_msoa_mapper.lsoa_to_msoa(lsoa)
            )
            
            # Remove rows where MSOA mapping failed
            original_count = len(df)
            df = df.dropna(subset=['msoa_code'])
            mapped_count = len(df)
            
            if original_count != mapped_count:
                print(f"⚠️ Mapped {mapped_count}/{original_count} LSOAs to MSOAs")
            
            # Aggregate LSOA data to MSOA level
            # For IMD decile, we'll use the median (most representative)
            # For IMD rank, we'll use the average
            msoa_aggregated = df.groupby('msoa_code').agg({
                'imd_decile': 'median',  # Use median decile for MSOA
                'imd_rank': 'mean',      # Use average rank for MSOA
                'la_code': 'first',      # Take first LA code (should be same for all LSOAs in MSOA)
                'la_name': 'first',      # Take first LA name
                'lsoa_code': 'count'     # Count of LSOAs in this MSOA
            }).reset_index()
            
            # Rename columns for clarity
            msoa_aggregated = msoa_aggregated.rename(columns={
                'imd_decile': 'msoa_imd_decile',
                'imd_rank': 'msoa_imd_rank',
                'lsoa_code': 'lsoa_count'
            })
            
            # Round the rank to integer
            msoa_aggregated['msoa_imd_rank'] = msoa_aggregated['msoa_imd_rank'].round().astype(int)
            
            print(f"✅ IMD data aggregated to {len(msoa_aggregated)} MSOAs")
            return msoa_aggregated
            
        except Exception as e:
            print(f"❌ Error loading IMD data: {e}")
            return None
    
    def _load_good_neighbours_data(self) -> Optional[pd.DataFrame]:
        """Load Good Neighbours data from Excel file"""
        try:
            file_path = self.data_config['good_neighbours']['file_path']
            if not os.path.exists(file_path):
                print(f"❌ Good Neighbours file not found: {file_path}")
                return None
            
            df = pd.read_excel(file_path, sheet_name=self.data_config['good_neighbours']['sheet_name'])
            
            # Standardize column names
            column_mapping = {
                'MSOA_code': 'msoa_code',
                'MSOA_name': 'msoa_name',
                'always_trust OR usually_trust': 'always_usually_trust',
                'usually_careful OR almost_always_careful': 'usually_almost_always_careful',
                'Net_trust': 'net_trust'
            }
            
            df = df.rename(columns=column_mapping)
            return df
            
        except Exception as e:
            print(f"❌ Error loading Good Neighbours data: {e}")
            return None
    
    def _load_population_data(self) -> Optional[pd.DataFrame]:
        """Load population data from Excel file"""
        try:
            file_path = self.data_config['population']['file_path']
            if not os.path.exists(file_path):
                print(f"❌ Population data file not found: {file_path}")
                return None
            
            # Load the data - try different header rows since Excel structure may vary
            df = pd.read_excel(file_path, sheet_name=self.data_config['population']['sheet_name'], header=None)
            print(f"✅ Loaded population data: {len(df)} rows")
            
            # Find the header row by looking for expected column names
            header_row = None
            for i in range(min(10, len(df))):  # Check first 10 rows
                row_values = df.iloc[i].astype(str).tolist()
                if any('LSOA' in str(val) and 'Code' in str(val) for val in row_values):
                    header_row = i
                    break
            
            if header_row is not None:
                # Use the found header row
                df = pd.read_excel(file_path, sheet_name=self.data_config['population']['sheet_name'], header=header_row)
                print(f"✅ Found headers in row {header_row + 1}")
            else:
                # Fallback: try with header=0
                df = pd.read_excel(file_path, sheet_name=self.data_config['population']['sheet_name'])
                print("⚠️ Using default header row")
            
            print(f"📋 Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            
            # Standardize column names - be flexible with column name variations
            column_mapping = {}
            
            # Find columns by partial matching
            for col in df.columns:
                col_str = str(col).lower()
                if 'lad' in col_str and 'code' in col_str:
                    column_mapping[col] = 'lad_code'
                elif 'lad' in col_str and 'name' in col_str:
                    column_mapping[col] = 'lad_name'
                elif 'lsoa' in col_str and 'code' in col_str:
                    column_mapping[col] = 'lsoa_code'
                elif 'lsoa' in col_str and 'name' in col_str:
                    column_mapping[col] = 'lsoa_name'
                elif col_str == 'total' or ('total' in col_str and 'population' in col_str):
                    column_mapping[col] = 'total_population'
            
            print(f"🔄 Column mapping: {column_mapping}")
            
            # Identify demographic columns (F0, M0, F1, M1, etc.) BEFORE renaming
            demographic_columns = [col for col in df.columns if col.startswith(('F', 'M')) and col[1:].isdigit()]
            self.demographic_columns = demographic_columns
            
            df = df.rename(columns=column_mapping)
            
            print(f"📊 Found {len(demographic_columns)} demographic columns")
            
            # Clean and validate data - check if required columns exist
            required_columns = ['lsoa_code', 'total_population']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing required columns after renaming: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            df = df.dropna(subset=['lsoa_code', 'total_population'])
            df['total_population'] = pd.to_numeric(df['total_population'], errors='coerce')
            
            # Convert demographic columns to numeric
            for col in demographic_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            print(f"✅ Population data processed: {len(df)} valid LSOAs")
            return df
            
        except Exception as e:
            print(f"❌ Error loading population data: {e}")
            return None
    
    def _aggregate_population_to_msoa(self):
        """Aggregate LSOA population data to MSOA level"""
        try:
            if self.population_data is None:
                return
            
            # Ensure LSOA-MSOA mapping is loaded
            if not lsoa_msoa_mapper.load_mapping_data():
                print("❌ Failed to load LSOA-MSOA mapping")
                return
            
            # Get MSOA mapping
            msoa_mapping = lsoa_msoa_mapper.lsoa_msoa_mapping
            
            # Debug: Check sample LSOA codes from population data
            sample_pop_lsoas = self.population_data['lsoa_code'].head(5).tolist()
            print(f"🔍 Sample LSOA codes from population data: {sample_pop_lsoas}")
            
            # Debug: Check sample LSOA codes from mapping
            sample_mapping_lsoas = list(msoa_mapping.keys())[:5]
            print(f"🔍 Sample LSOA codes from mapping: {sample_mapping_lsoas}")
            
            # Add MSOA codes to population data
            self.population_data['msoa_code'] = self.population_data['lsoa_code'].map(msoa_mapping)
            
            # Remove LSOAs without MSOA mapping
            valid_data = self.population_data.dropna(subset=['msoa_code'])
            print(f"📊 Mapped {len(valid_data)}/{len(self.population_data)} LSOAs to MSOAs")
            
            # Aggregate by MSOA
            agg_columns = ['total_population'] + self.demographic_columns
            msoa_agg = valid_data.groupby('msoa_code')[agg_columns].sum().reset_index()
            
            # Add MSOA names (take the first LAD name for each MSOA)
            msoa_names = valid_data.groupby('msoa_code')['lad_name'].first().reset_index()
            msoa_names.columns = ['msoa_code', 'msoa_name']
            
            self.msoa_population_data = msoa_agg.merge(msoa_names, on='msoa_code', how='left')
            
            print(f"✅ Population data aggregated to {len(self.msoa_population_data)} MSOAs")
            
        except Exception as e:
            print(f"❌ Error aggregating population data: {e}")
    
    def get_msoa_data(self, msoa_code: str, data_sources: List[str] = None) -> Dict[str, MSOADataResult]:
        """
        Get comprehensive data for an MSOA from multiple sources
        
        Args:
            msoa_code: MSOA code to query
            data_sources: List of data sources to query (default: all available)
            
        Returns:
            Dictionary mapping data source names to MSOADataResult objects
        """
        if data_sources is None:
            data_sources = ['imd', 'good_neighbours']
        
        results = {}
        
        for source in data_sources:
            if source == 'imd':
                results[source] = self._get_imd_data(msoa_code)
            elif source == 'good_neighbours':
                results[source] = self._get_good_neighbours_data(msoa_code)
            else:
                results[source] = MSOADataResult(
                    msoa_code=msoa_code,
                    msoa_name="Unknown",
                    data_source=source,
                    data={},
                    timestamp=datetime.now(),
                    success=False,
                    error_message=f"Unknown data source: {source}"
                )
        
        return results
    
    def _get_imd_data(self, msoa_code: str) -> MSOADataResult:
        """Get IMD data for an MSOA"""
        try:
            if self.imd_data is None:
                return MSOADataResult(
                    msoa_code=msoa_code,
                    msoa_name="Unknown",
                    data_source="imd",
                    data={},
                    timestamp=datetime.now(),
                    success=False,
                    error_message="IMD data not loaded"
                )
            
            # Filter data for the MSOA (now already aggregated)
            msoa_data = self.imd_data[self.imd_data['msoa_code'] == msoa_code]
            
            if msoa_data.empty:
                return MSOADataResult(
                    msoa_code=msoa_code,
                    msoa_name="Unknown",
                    data_source="imd",
                    data={},
                    timestamp=datetime.now(),
                    success=False,
                    error_message=f"No IMD data found for MSOA {msoa_code}"
                )
            
            # Extract aggregated data for the MSOA
            row = msoa_data.iloc[0]
            aggregated_data = {
                'lsoa_count': int(row.get('lsoa_count', 0)),
                'msoa_imd_rank': int(row.get('msoa_imd_rank', 0)),
                'msoa_imd_decile': float(row.get('msoa_imd_decile', 0)),
                'la_code': row.get('la_code', 'Unknown'),
                'la_name': row.get('la_name', 'Unknown'),
                'deprivation_level': self._get_deprivation_level(row.get('msoa_imd_decile', 0))
            }
            
            return MSOADataResult(
                msoa_code=msoa_code,
                msoa_name=f"MSOA {msoa_code}",
                data_source="imd",
                data=aggregated_data,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            return MSOADataResult(
                msoa_code=msoa_code,
                msoa_name="Unknown",
                data_source="imd",
                data={},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def _get_good_neighbours_data(self, msoa_code: str) -> MSOADataResult:
        """Get Good Neighbours data for an MSOA"""
        try:
            if self.good_neighbours_data is None:
                return MSOADataResult(
                    msoa_code=msoa_code,
                    msoa_name="Unknown",
                    data_source="good_neighbours",
                    data={},
                    timestamp=datetime.now(),
                    success=False,
                    error_message="Good Neighbours data not loaded"
                )
            
            # Filter data for the MSOA
            msoa_data = self.good_neighbours_data[self.good_neighbours_data['msoa_code'] == msoa_code]
            
            if msoa_data.empty:
                return MSOADataResult(
                    msoa_code=msoa_code,
                    msoa_name="Unknown",
                    data_source="good_neighbours",
                    data={},
                    timestamp=datetime.now(),
                    success=False,
                    error_message=f"No Good Neighbours data found for MSOA {msoa_code}"
                )
            
            # Extract data
            row = msoa_data.iloc[0]
            data = {
                'msoa_name': row.get('msoa_name', 'Unknown'),
                'always_usually_trust': row.get('always_usually_trust', 0),
                'usually_almost_always_careful': row.get('usually_almost_always_careful', 0),
                'net_trust': row.get('net_trust', 0)
            }
            
            return MSOADataResult(
                msoa_code=msoa_code,
                msoa_name=data['msoa_name'],
                data_source="good_neighbours",
                data=data,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            return MSOADataResult(
                msoa_code=msoa_code,
                msoa_name="Unknown",
                data_source="good_neighbours",
                data={},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def get_multiple_msoas_data(self, msoa_codes: List[str], data_sources: List[str] = None) -> Dict[str, Dict[str, MSOADataResult]]:
        """
        Get data for multiple MSOAs
        
        Args:
            msoa_codes: List of MSOA codes
            data_sources: List of data sources to query
            
        Returns:
            Dictionary mapping MSOA codes to data source results
        """
        results = {}
        for msoa_code in msoa_codes:
            results[msoa_code] = self.get_msoa_data(msoa_code, data_sources)
        return results
    
    def get_top_performing_msoas(self, metric: str, n: int = 10, data_source: str = 'good_neighbours') -> List[Dict[str, Any]]:
        """
        Get top performing MSOAs by a specific metric
        
        Args:
            metric: Metric to rank by (e.g., 'net_trust', 'imd_decile')
            n: Number of top MSOAs to return
            data_source: Data source to query
            
        Returns:
            List of top performing MSOAs
        """
        try:
            if data_source == 'good_neighbours' and self.good_neighbours_data is not None:
                # Sort by metric (descending for net_trust, ascending for decile)
                ascending = metric in ['imd_decile', 'imd_rank']
                top_data = self.good_neighbours_data.nlargest(n, metric) if not ascending else self.good_neighbours_data.nsmallest(n, metric)
                
                results = []
                for _, row in top_data.iterrows():
                    results.append({
                        'msoa_code': row['msoa_code'],
                        'msoa_name': row['msoa_name'],
                        'metric_value': row[metric],
                        'always_usually_trust': row.get('always_usually_trust', 0),
                        'usually_almost_always_careful': row.get('usually_almost_always_careful', 0),
                        'net_trust': row.get('net_trust', 0)
                    })
                return results
            
            elif data_source == 'imd' and self.imd_data is not None:
                # For IMD, we need to aggregate by MSOA first
                msoa_aggregated = self.imd_data.groupby('msoa_code').agg({
                    'imd_rank': 'mean',
                    'imd_decile': 'mean',
                    'la_name': 'first'
                }).reset_index()
                
                ascending = metric in ['imd_decile', 'imd_rank']
                top_data = msoa_aggregated.nlargest(n, metric) if not ascending else msoa_aggregated.nsmallest(n, metric)
                
                results = []
                for _, row in top_data.iterrows():
                    results.append({
                        'msoa_code': row['msoa_code'],
                        'msoa_name': f"MSOA {row['msoa_code']}",
                        'metric_value': row[metric],
                        'imd_rank': row['imd_rank'],
                        'imd_decile': row['imd_decile'],
                        'la_name': row['la_name']
                    })
                return results
            
            return []
            
        except Exception as e:
            print(f"Error getting top performing MSOAs: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data sources"""
        summary = {
            'data_sources': {},
            'total_msoas': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        if self.imd_data is not None:
            unique_msoas = self.imd_data['msoa_code'].nunique()
            summary['data_sources']['imd'] = {
                'loaded': True,
                'total_records': len(self.imd_data),
                'unique_msoas': unique_msoas,
                'columns': list(self.imd_data.columns)
            }
            summary['total_msoas'] = max(summary['total_msoas'], unique_msoas)
        else:
            summary['data_sources']['imd'] = {'loaded': False}
        
        if self.good_neighbours_data is not None:
            unique_msoas = self.good_neighbours_data['msoa_code'].nunique()
            summary['data_sources']['good_neighbours'] = {
                'loaded': True,
                'total_records': len(self.good_neighbours_data),
                'unique_msoas': unique_msoas,
                'columns': list(self.good_neighbours_data.columns)
            }
            summary['total_msoas'] = max(summary['total_msoas'], unique_msoas)
        else:
            summary['data_sources']['good_neighbours'] = {'loaded': False}
        
        return summary
    
    def export_data(self, msoa_codes: List[str] = None, format: str = 'json') -> str:
        """
        Export data for specified MSOAs
        
        Args:
            msoa_codes: List of MSOA codes (None for all)
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data as string
        """
        if msoa_codes is None:
            # Export all available MSOAs
            if self.good_neighbours_data is not None:
                msoa_codes = self.good_neighbours_data['msoa_code'].unique().tolist()
            elif self.imd_data is not None:
                msoa_codes = self.imd_data['msoa_code'].unique().tolist()
            else:
                return "No data available for export"
        
        # Get data for all MSOAs
        all_data = self.get_multiple_msoas_data(msoa_codes)
        
        if format == 'json':
            # Convert to JSON-serializable format
            json_data = {}
            for msoa_code, sources in all_data.items():
                json_data[msoa_code] = {}
                for source, result in sources.items():
                    json_data[msoa_code][source] = {
                        'success': result.success,
                        'data': result.data,
                        'timestamp': result.timestamp.isoformat(),
                        'error_message': result.error_message
                    }
            return json.dumps(json_data, indent=2)
        
        elif format == 'csv':
            # Flatten data for CSV
            rows = []
            for msoa_code, sources in all_data.items():
                row = {'msoa_code': msoa_code}
                for source, result in sources.items():
                    if result.success:
                        for key, value in result.data.items():
                            row[f"{source}_{key}"] = value
                    else:
                        row[f"{source}_error"] = result.error_message
                rows.append(row)
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reload_data(self):
        """Reload all data sources"""
        print("🔄 Reloading data sources...")
        self._load_data_sources()
    
    def _get_deprivation_level(self, decile: float) -> str:
        """Convert IMD decile to deprivation level description"""
        if decile <= 2:
            return "Most Deprived"
        elif decile <= 4:
            return "Very Deprived"
        elif decile <= 6:
            return "Moderately Deprived"
        elif decile <= 8:
            return "Less Deprived"
        else:
            return "Least Deprived"
    
    def load_good_neighbours_data(self) -> Optional[pd.DataFrame]:
        """Load Good Neighbours data (public method for dashboard compatibility)"""
        return self.good_neighbours_data
    
    def get_good_neighbours_summary(self) -> Optional[Dict[str, Any]]:
        """Get Good Neighbours data summary (public method for dashboard compatibility)"""
        if self.good_neighbours_data is None:
            return None
        
        try:
            # Calculate trust distribution
            positive_trust = len(self.good_neighbours_data[self.good_neighbours_data['net_trust'] > 0])
            negative_trust = len(self.good_neighbours_data[self.good_neighbours_data['net_trust'] < 0])
            neutral_trust = len(self.good_neighbours_data[self.good_neighbours_data['net_trust'] == 0])
            
            summary = {
                'total_msoas': len(self.good_neighbours_data),
                'average_net_trust': self.good_neighbours_data['net_trust'].mean(),  # Dashboard expects this key
                'avg_net_trust': self.good_neighbours_data['net_trust'].mean(),  # Keep for compatibility
                'min_net_trust': self.good_neighbours_data['net_trust'].min(),
                'max_net_trust': self.good_neighbours_data['net_trust'].max(),
                'std_net_trust': self.good_neighbours_data['net_trust'].std(),
                'avg_always_usually_trust': self.good_neighbours_data['always_usually_trust'].mean(),
                'avg_usually_almost_always_careful': self.good_neighbours_data['usually_almost_always_careful'].mean(),
                'net_trust_distribution': {  # Dashboard expects this structure
                    'positive_trust': positive_trust,
                    'negative_trust': negative_trust,
                    'neutral_trust': neutral_trust
                }
            }
            return summary
        except Exception as e:
            print(f"❌ Error creating Good Neighbours summary: {e}")
            return None
    
    def get_population_data(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """Get population data for a specific MSOA"""
        if self.msoa_population_data is None:
            return None
        
        try:
            row = self.msoa_population_data[self.msoa_population_data['msoa_code'] == msoa_code].iloc[0]
            
            # Extract demographics
            demographics = {}
            for col in self.demographic_columns:
                demographics[col] = int(row[col])
            
            return {
                'msoa_code': row['msoa_code'],
                'msoa_name': row['msoa_name'],
                'total_population': int(row['total_population']),
                'demographics': demographics,
                'timestamp': datetime.now()
            }
            
        except (IndexError, KeyError):
            return None
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get overall population statistics"""
        if self.msoa_population_data is None:
            return {}
        
        try:
            total_pop = self.msoa_population_data['total_population'].sum()
            avg_pop = self.msoa_population_data['total_population'].mean()
            min_pop = self.msoa_population_data['total_population'].min()
            max_pop = self.msoa_population_data['total_population'].max()
            
            # Calculate age group distributions
            age_groups = self._calculate_age_groups()
            
            return {
                'total_population': total_pop,
                'total_msoas': len(self.msoa_population_data),
                'average_population_per_msoa': avg_pop,
                'min_population': min_pop,
                'max_population': max_pop,
                'age_groups': age_groups,
                'demographic_columns': len(self.demographic_columns)
            }
            
        except Exception as e:
            print(f"❌ Error creating population summary: {e}")
            return {}
    
    def get_demographic_analysis(self, msoa_code: str = None) -> Dict[str, Any]:
        """Get demographic analysis for a specific MSOA or overall"""
        if self.msoa_population_data is None:
            return {}
        
        try:
            if msoa_code:
                data = self.msoa_population_data[self.msoa_population_data['msoa_code'] == msoa_code]
                if data.empty:
                    return {}
                data = data.iloc[0]
            else:
                data = self.msoa_population_data.sum()
            
            # Calculate gender distribution
            female_total = sum(data[col] for col in self.demographic_columns if col.startswith('F'))
            male_total = sum(data[col] for col in self.demographic_columns if col.startswith('M'))
            
            # Calculate age distribution
            age_groups = self._calculate_age_groups_for_data(data)
            
            return {
                'total_population': int(data['total_population']),
                'gender_distribution': {
                    'female': int(female_total),
                    'male': int(male_total),
                    'female_percentage': (female_total / data['total_population']) * 100,
                    'male_percentage': (male_total / data['total_population']) * 100
                },
                'age_groups': age_groups
            }
            
        except Exception as e:
            print(f"❌ Error in demographic analysis: {e}")
            return {}
    
    def get_top_populated_msoas(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most populated MSOAs"""
        if self.msoa_population_data is None:
            return []
        
        try:
            top_msoas = self.msoa_population_data.nlargest(n, 'total_population')
            
            result = []
            for _, row in top_msoas.iterrows():
                result.append({
                    'msoa_code': row['msoa_code'],
                    'msoa_name': row['msoa_name'],
                    'total_population': int(row['total_population']),
                    'rank': len(result) + 1
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting top populated MSOAs: {e}")
            return []
    
    def _calculate_age_groups(self) -> Dict[str, int]:
        """Calculate population by age groups"""
        if self.msoa_population_data is None:
            return {}
        
        try:
            age_groups = {
                '0-4': 0, '5-9': 0, '10-14': 0, '15-19': 0, '20-24': 0,
                '25-29': 0, '30-34': 0, '35-39': 0, '40-44': 0, '45-49': 0,
                '50-54': 0, '55-59': 0, '60-64': 0, '65-69': 0, '70-74': 0,
                '75-79': 0, '80-84': 0, '85+': 0
            }
            
            # Sum up age groups
            for col in self.demographic_columns:
                if col.startswith('F') or col.startswith('M'):
                    age = int(col[1:])
                    total_for_age = self.msoa_population_data[col].sum()
                    
                    # Map to age groups
                    if age <= 4:
                        age_groups['0-4'] += total_for_age
                    elif age <= 9:
                        age_groups['5-9'] += total_for_age
                    elif age <= 14:
                        age_groups['10-14'] += total_for_age
                    elif age <= 19:
                        age_groups['15-19'] += total_for_age
                    elif age <= 24:
                        age_groups['20-24'] += total_for_age
                    elif age <= 29:
                        age_groups['25-29'] += total_for_age
                    elif age <= 34:
                        age_groups['30-34'] += total_for_age
                    elif age <= 39:
                        age_groups['35-39'] += total_for_age
                    elif age <= 44:
                        age_groups['40-44'] += total_for_age
                    elif age <= 49:
                        age_groups['45-49'] += total_for_age
                    elif age <= 54:
                        age_groups['50-54'] += total_for_age
                    elif age <= 59:
                        age_groups['55-59'] += total_for_age
                    elif age <= 64:
                        age_groups['60-64'] += total_for_age
                    elif age <= 69:
                        age_groups['65-69'] += total_for_age
                    elif age <= 74:
                        age_groups['70-74'] += total_for_age
                    elif age <= 79:
                        age_groups['75-79'] += total_for_age
                    elif age <= 84:
                        age_groups['80-84'] += total_for_age
                    else:
                        age_groups['85+'] += total_for_age
            
            return age_groups
            
        except Exception as e:
            print(f"❌ Error calculating age groups: {e}")
            return {}
    
    def _calculate_age_groups_for_data(self, data) -> Dict[str, int]:
        """Calculate age groups for specific data row"""
        age_groups = {}
        
        for col in self.demographic_columns:
            if col.startswith('F') or col.startswith('M'):
                age = int(col[1:])
                total_for_age = int(data[col])
                
                # Map to age groups
                if age <= 4:
                    age_groups['0-4'] = age_groups.get('0-4', 0) + total_for_age
                elif age <= 9:
                    age_groups['5-9'] = age_groups.get('5-9', 0) + total_for_age
                elif age <= 14:
                    age_groups['10-14'] = age_groups.get('10-14', 0) + total_for_age
                elif age <= 19:
                    age_groups['15-19'] = age_groups.get('15-19', 0) + total_for_age
                elif age <= 24:
                    age_groups['20-24'] = age_groups.get('20-24', 0) + total_for_age
                elif age <= 29:
                    age_groups['25-29'] = age_groups.get('25-29', 0) + total_for_age
                elif age <= 34:
                    age_groups['30-34'] = age_groups.get('30-34', 0) + total_for_age
                elif age <= 39:
                    age_groups['35-39'] = age_groups.get('35-39', 0) + total_for_age
                elif age <= 44:
                    age_groups['40-44'] = age_groups.get('40-44', 0) + total_for_age
                elif age <= 49:
                    age_groups['45-49'] = age_groups.get('45-49', 0) + total_for_age
                elif age <= 54:
                    age_groups['50-54'] = age_groups.get('50-54', 0) + total_for_age
                elif age <= 59:
                    age_groups['55-59'] = age_groups.get('55-59', 0) + total_for_age
                elif age <= 64:
                    age_groups['60-64'] = age_groups.get('60-64', 0) + total_for_age
                elif age <= 69:
                    age_groups['65-69'] = age_groups.get('65-69', 0) + total_for_age
                elif age <= 74:
                    age_groups['70-74'] = age_groups.get('70-74', 0) + total_for_age
                elif age <= 79:
                    age_groups['75-79'] = age_groups.get('75-79', 0) + total_for_age
                elif age <= 84:
                    age_groups['80-84'] = age_groups.get('80-84', 0) + total_for_age
                else:
                    age_groups['85+'] = age_groups.get('85+', 0) + total_for_age
        
        return age_groups
    
    def _load_population_from_cache(self) -> bool:
        """Load pre-aggregated MSOA population data from cache"""
        try:
            print(f"🔍 Checking cache file: {self.population_cache_file}")
            
            if not os.path.exists(self.population_cache_file):
                print("❌ Cache file does not exist")
                return False
            
            print(f"✅ Cache file exists: {os.path.getsize(self.population_cache_file)} bytes")
            
            # Check if cache is newer than source file
            source_file = self.data_config['population']['file_path']
            if os.path.exists(source_file):
                cache_time = os.path.getmtime(self.population_cache_file)
                source_time = os.path.getmtime(source_file)
                if source_time > cache_time:
                    print("🔄 Population source file is newer than cache, will reload")
                    return False
            
            # Load cached data
            print("📖 Loading cached data...")
            with open(self.population_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Reconstruct MSOA population data
            self.msoa_population_data = pd.DataFrame(cache_data['msoa_data'])
            self.demographic_columns = cache_data['demographic_columns']
            
            print(f"✅ Loaded {len(self.msoa_population_data)} MSOAs from cache")
            return True
            
        except Exception as e:
            print(f"❌ Error loading population cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_population_cache(self):
        """Save pre-aggregated MSOA population data to cache"""
        try:
            if self.msoa_population_data is None:
                print("⚠️ No MSOA population data to cache")
                return
            
            print(f"💾 Saving population cache to: {self.population_cache_file}")
            
            cache_data = {
                'msoa_data': self.msoa_population_data.to_dict('records'),
                'demographic_columns': self.demographic_columns,
                'timestamp': datetime.now().isoformat(),
                'total_msoas': len(self.msoa_population_data)
            }
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.population_cache_file), exist_ok=True)
            
            with open(self.population_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"✅ Saved population cache: {len(self.msoa_population_data)} MSOAs")
            print(f"📁 Cache file size: {os.path.getsize(self.population_cache_file)} bytes")
            
        except Exception as e:
            print(f"❌ Error saving population cache: {e}")
            import traceback
            traceback.print_exc()
    
    def refresh_population_cache(self):
        """Force refresh of population cache"""
        print("🔄 Refreshing population cache...")
        
        # Remove existing cache file if it exists
        if os.path.exists(self.population_cache_file):
            os.remove(self.population_cache_file)
            print("🗑️ Removed existing cache file")
        
        # Force reload population data (bypass cache check)
        print("📊 Loading population data from source...")
        self.population_data = self._load_population_data()
        if self.population_data is not None:
            print("🔄 Aggregating to MSOA level...")
            self._aggregate_population_to_msoa()
            print("💾 Saving cache...")
            self._save_population_cache()
            print("✅ Population cache refreshed successfully")
        else:
            print("❌ Failed to refresh population cache")
    
    def _load_community_survey_data(self) -> Optional[pd.DataFrame]:
        """Load Community Life Survey data"""
        try:
            if not self.data_config['community_survey']['enabled']:
                return None
            
            print("🔄 Loading Community Life Survey data...")
            
            # Process all sheets from the Community Life Survey
            processed_data = self.community_survey_connector.process_all_sheets()
            
            if processed_data.empty:
                print("❌ No Community Life Survey data could be processed")
                return None
            
            print(f"✅ Community Life Survey data processed: {len(processed_data)} responses")
            return processed_data
            
        except Exception as e:
            print(f"❌ Error loading Community Life Survey data: {e}")
            return None
    
    def get_community_survey_data(self) -> Optional[pd.DataFrame]:
        """Get Community Life Survey data"""
        return self.community_survey_data
    
    def _load_unemployment_data(self) -> Optional[pd.DataFrame]:
        """Load unemployment data"""
        try:
            if self.unemployment_connector is None:
                print("⚠️ Unemployment connector not available, skipping unemployment data loading")
                return None
                
            print("🔄 Loading unemployment data...")
            
            # Load data using the unemployment connector
            unemployment_data = self.unemployment_connector.get_unemployment_data()
            
            if unemployment_data is None or unemployment_data.empty:
                print("❌ No unemployment data could be loaded")
                return None
            
            print(f"✅ Unemployment data loaded: {len(unemployment_data)} areas")
            return unemployment_data
            
        except Exception as e:
            print(f"❌ Error loading unemployment data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_unemployment_data(self) -> Optional[pd.DataFrame]:
        """Get unemployment data"""
        return self.unemployment_data
    
    def get_unemployment_by_lad(self, lad_name: str) -> Optional[Dict[str, Any]]:
        """Get unemployment data for a specific LAD"""
        if self.unemployment_connector is None:
            print(f"⚠️ Warning: Unemployment connector not available")
            return None
        try:
            return self.unemployment_connector.get_unemployment_by_lad(lad_name)
        except Exception as e:
            print(f"❌ Error getting unemployment data for {lad_name}: {e}")
            return None
    
    def get_unemployment_summary(self) -> Dict[str, Any]:
        """Get summary of unemployment data"""
        if self.unemployment_data is None:
            return {}
        
        if self.unemployment_connector is None:
            return {}
            
        try:
            return self.unemployment_connector.get_unemployment_summary()
        except Exception as e:
            print(f"❌ Error getting unemployment summary: {e}")
            return {}
    
    def get_community_survey_summary(self) -> Dict[str, Any]:
        """Get summary of Community Life Survey data"""
        if self.community_survey_data is None:
            return {}
        
        summary = self.community_survey_connector.get_question_summary()
        
        # Add LAD-level summary
        if not self.community_survey_data.empty:
            lad_column = self.community_survey_data.columns[1]  # Column B should be LAD names
            unique_lads = self.community_survey_data[lad_column].nunique()
            summary['unique_lads'] = unique_lads
            summary['total_responses'] = len(self.community_survey_data)
        
        return summary
    
    def get_lad_survey_data(self, lad_name: str) -> pd.DataFrame:
        """Get Community Life Survey data for a specific Local Authority District"""
        if self.community_survey_data is None:
            return pd.DataFrame()
        
        # Filter data for the specific LAD
        lad_column = self.community_survey_data.columns[1]  # Column B should be LAD names
        lad_data = self.community_survey_data[self.community_survey_data[lad_column] == lad_name]
        return lad_data
    
    def get_survey_question_data(self, question: str) -> pd.DataFrame:
        """Get Community Life Survey data for a specific question"""
        if self.community_survey_data is None:
            return pd.DataFrame()
        
        # Filter data for the specific question
        question_data = self.community_survey_data[self.community_survey_data['question'] == question]
        return question_data
    
    def get_top_survey_questions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most common survey questions"""
        if self.community_survey_data is None:
            return []
        
        question_counts = self.community_survey_data['question'].value_counts()
        
        top_questions = []
        for i, (question, count) in enumerate(question_counts.head(n).items(), 1):
            top_questions.append({
                'rank': i,
                'question': question,
                'response_count': count
            })
        
        return top_questions
    
    def get_all_survey_questions(self) -> List[str]:
        """Get all unique survey questions"""
        if self.community_survey_data is None:
            return []
        
        unique_questions = self.community_survey_data['question'].unique()
        return sorted(unique_questions.tolist())
    
    def refresh_community_survey_data(self):
        """Refresh Community Life Survey data to pick up updated question text"""
        if self.community_survey_connector:
            print("🔄 Refreshing Community Life Survey data...")
            # Force reload by clearing the cache
            self.community_survey_connector._data_loaded = False
            self.community_survey_connector.survey_data = {}
            # Reload all sheets
            self.community_survey_connector.load_all_sheets()
            # Process the data to extract question text
            self.community_survey_data = self.community_survey_connector.process_all_sheets()
            print(f"✅ Community Life Survey data refreshed: {len(self.community_survey_data)} rows")
