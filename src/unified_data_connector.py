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
    
    def __init__(self):
        self.data_config = self._load_data_config()
        self.imd_data = None
        self.good_neighbours_data = None
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
                
        except Exception as e:
            print(f"❌ Error loading data sources: {e}")
    
    def _load_imd_data(self) -> Optional[pd.DataFrame]:
        """Load IMD data from Excel file"""
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
            
            # Extract MSOA code from LSOA code (first 9 characters)
            df['msoa_code'] = df['lsoa_code'].str[:9]
            
            return df
            
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
            
            # Filter data for the MSOA
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
            
            # Aggregate data for the MSOA
            aggregated_data = {
                'total_lsoas': len(msoa_data),
                'imd_rank': msoa_data['imd_rank'].mean(),
                'imd_decile': msoa_data['imd_decile'].mean(),
                'la_name': msoa_data['la_name'].iloc[0] if 'la_name' in msoa_data.columns else 'Unknown',
                'lsoa_codes': msoa_data['lsoa_code'].tolist(),
                'lsoa_names': msoa_data['lsoa_name'].tolist()
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
        print("✅ Data reload complete")
