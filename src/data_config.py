#!/usr/bin/env python3
"""
Data Configuration System
Centralized configuration for switching between dummy and real data sources
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataSourceConfig:
    """Configuration for a specific data source"""
    enabled: bool
    use_real_data: bool
    fallback_to_dummy: bool
    data_url: Optional[str] = None
    local_file_path: Optional[str] = None
    api_key: Optional[str] = None
    update_frequency_days: int = 30

class DataConfig:
    """
    Centralized data configuration system
    Controls whether to use real data sources or dummy data
    """
    
    def __init__(self):
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> Dict[str, DataSourceConfig]:
        """Load configuration from environment variables and defaults"""
        
        # Default configuration - all dummy data
        default_config = {
            'imd_data': DataSourceConfig(
                enabled=True,
                use_real_data=True,  # Enable real IMD data since we have the file
                fallback_to_dummy=True,
                data_url="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833970/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx",
                local_file_path="data/IMD2019_Index_of_Multiple_Deprivation.xlsx"
            ),
            'ons_census': DataSourceConfig(
                enabled=True,
                use_real_data=False,
                fallback_to_dummy=True,
                data_url="https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/middlesuperoutputareamidyearpopulationestimates/mid2020/sape22dt1amid2020on2021geography.xlsx",
                local_file_path="data/census_data.xlsx"
            ),
            'community_life_survey': DataSourceConfig(
                enabled=True,
                use_real_data=False,
                fallback_to_dummy=True,
                data_url="https://www.gov.uk/government/statistics/community-life-survey-2020-21",
                local_file_path="data/community_life_survey.xlsx"
            ),
            'crime_data': DataSourceConfig(
                enabled=True,
                use_real_data=False,
                fallback_to_dummy=True,
                data_url="https://data.police.uk/data/",
                local_file_path="data/crime_data.csv"
            ),
            'economic_data': DataSourceConfig(
                enabled=True,
                use_real_data=False,
                fallback_to_dummy=True,
                data_url="https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/employmentbyoccupationemp04",
                local_file_path="data/economic_data.xlsx"
            ),
            'postcode_data': DataSourceConfig(
                enabled=True,
                use_real_data=True,  # Always use real postcode data
                fallback_to_dummy=False,
                data_url="https://api.postcodes.io/",
                local_file_path=None
            )
        }
        
        # Override with environment variables if present
        for source_name, config in default_config.items():
            env_prefix = source_name.upper()
            
            # Check for real data flag
            real_data_env = f"{env_prefix}_USE_REAL_DATA"
            if real_data_env in os.environ:
                config.use_real_data = os.environ[real_data_env].lower() in ('true', '1', 'yes')
            
            # Check for enabled flag
            enabled_env = f"{env_prefix}_ENABLED"
            if enabled_env in os.environ:
                config.enabled = os.environ[enabled_env].lower() in ('true', '1', 'yes')
            
            # Check for custom URL
            url_env = f"{env_prefix}_URL"
            if url_env in os.environ:
                config.data_url = os.environ[url_env]
            
            # Check for local file path
            file_env = f"{env_prefix}_FILE_PATH"
            if file_env in os.environ:
                config.local_file_path = os.environ[file_env]
            
            # Check for API key
            key_env = f"{env_prefix}_API_KEY"
            if key_env in os.environ:
                config.api_key = os.environ[key_env]
        
        return default_config
    
    def get_config(self, source_name: str) -> DataSourceConfig:
        """Get configuration for a specific data source"""
        return self.config.get(source_name, DataSourceConfig(
            enabled=False,
            use_real_data=False,
            fallback_to_dummy=True
        ))
    
    def use_real_data(self, source_name: str) -> bool:
        """Check if real data should be used for a source"""
        config = self.get_config(source_name)
        return config.enabled and config.use_real_data
    
    def use_dummy_data(self, source_name: str) -> bool:
        """Check if dummy data should be used for a source"""
        config = self.get_config(source_name)
        return not config.use_real_data or (config.fallback_to_dummy and not self._real_data_available(source_name))
    
    def _real_data_available(self, source_name: str) -> bool:
        """Check if real data is available for a source"""
        config = self.get_config(source_name)
        
        # Check if local file exists
        if config.local_file_path and os.path.exists(config.local_file_path):
            return True
        
        # Check if we can download from URL
        if config.data_url:
            try:
                import requests
                response = requests.head(config.data_url, timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False
    
    def get_data_url(self, source_name: str) -> Optional[str]:
        """Get data URL for a source"""
        return self.get_config(source_name).data_url
    
    def get_local_file_path(self, source_name: str) -> Optional[str]:
        """Get local file path for a source"""
        return self.get_config(source_name).local_file_path
    
    def get_api_key(self, source_name: str) -> Optional[str]:
        """Get API key for a source"""
        return self.get_config(source_name).api_key
    
    def list_enabled_sources(self) -> list:
        """List all enabled data sources"""
        return [name for name, config in self.config.items() if config.enabled]
    
    def list_real_data_sources(self) -> list:
        """List all sources configured to use real data"""
        return [name for name, config in self.config.items() 
                if config.enabled and config.use_real_data]
    
    def list_dummy_data_sources(self) -> list:
        """List all sources using dummy data"""
        return [name for name, config in self.config.items() 
                if config.enabled and self.use_dummy_data(name)]
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of data source status"""
        return {
            'total_sources': len(self.config),
            'enabled_sources': len(self.list_enabled_sources()),
            'real_data_sources': len(self.list_real_data_sources()),
            'dummy_data_sources': len(self.list_dummy_data_sources()),
            'sources': {
                name: {
                    'enabled': config.enabled,
                    'use_real_data': config.use_real_data,
                    'real_data_available': self._real_data_available(name),
                    'data_url': config.data_url,
                    'local_file_path': config.local_file_path
                }
                for name, config in self.config.items()
            }
        }

# Global configuration instance
data_config = DataConfig()

def get_data_config() -> DataConfig:
    """Get the global data configuration instance"""
    return data_config

def use_real_data(source_name: str) -> bool:
    """Convenience function to check if real data should be used"""
    return data_config.use_real_data(source_name)

def use_dummy_data(source_name: str) -> bool:
    """Convenience function to check if dummy data should be used"""
    return data_config.use_dummy_data(source_name)

def get_data_url(source_name: str) -> Optional[str]:
    """Convenience function to get data URL"""
    return data_config.get_data_url(source_name)

def get_local_file_path(source_name: str) -> Optional[str]:
    """Convenience function to get local file path"""
    return data_config.get_local_file_path(source_name)

def get_api_key(source_name: str) -> Optional[str]:
    """Convenience function to get API key"""
    return data_config.get_api_key(source_name)
