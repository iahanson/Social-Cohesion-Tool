#!/usr/bin/env python3
"""
Local News Data Connector

Handles loading and processing of local news data from scraped news agencies.
Includes location mapping for LADs and referenced places.
"""

import pandas as pd
import os
import re
from typing import Dict, List, Optional, Any, Tuple
import json

class LocalNewsConnector:
    """Connector for local news data with location mapping"""
    
    def __init__(self, data_file: str = "data/england_local_news_batch100_full_completed.csv"):
        self.data_file = data_file
        self.news_data = None
        self._data_loaded = False
        
        # LAD mapping cache
        self._lad_mapping_cache = None
        self._place_mapping_cache = None
        
        # Check if file exists
        if not os.path.exists(self.data_file):
            print(f"⚠️ Warning: Local news data file not found: {self.data_file}")
            print("   Available files in data directory:")
            try:
                data_files = [f for f in os.listdir("data") if f.endswith('.csv')]
                for file in data_files:
                    print(f"   - {file}")
            except:
                print("   Could not list data directory")
    
    def load_data(self) -> bool:
        """Load local news data from CSV file"""
        try:
            if not os.path.exists(self.data_file):
                print(f"❌ Local news data file not found: {self.data_file}")
                return False
            
            print(f"📰 Loading local news data from {self.data_file}")
            
            # Read the CSV file
            self.news_data = pd.read_csv(self.data_file)
            print(f"📰 News data loaded successfully: {self.news_data.shape[0]} articles, {self.news_data.shape[1]} columns")
            
            # Clean column names
            self.news_data.columns = [col.strip() for col in self.news_data.columns]
            print(f"📰 Available columns: {list(self.news_data.columns)}")
            
            # Clean the data
            self.news_data = self.news_data.dropna(subset=['local_authority_district', 'brief_description'])
            
            # Initialize location mapping
            self._initialize_location_mapping()
            
            self._data_loaded = True
            print(f"✅ Loaded local news data: {len(self.news_data)} articles")
            
            # Show sample data
            if len(self.news_data) > 0:
                print("📰 Sample news data:")
                print(self.news_data.head(2))
                
                # Test mapping for sample LADs
                print("📰 Testing LAD mapping:")
                sample_lads = self.news_data['local_authority_district'].head(5).tolist()
                for lad in sample_lads:
                    mapped = self._map_lad_to_standard(lad)
                    normalized = self._normalize_lad_name(lad)
                    print(f"   '{lad}' -> normalized: '{normalized}' -> mapped: '{mapped}'")
                
                # Test if LAD mapping cache is loaded
                if self._lad_mapping_cache is not None:
                    print(f"📰 LAD mapping cache loaded: {len(self._lad_mapping_cache)} LADs")
                    print(f"📰 Cache columns: {list(self._lad_mapping_cache.columns)}")
                    # Try different possible column names for LAD name
                    lad_col = 'LAD24NM' if 'LAD24NM' in self._lad_mapping_cache.columns else 'LAD23NM' if 'LAD23NM' in self._lad_mapping_cache.columns else 'LAD22NM'
                    sample_cache_lads = self._lad_mapping_cache[lad_col].head(5).tolist()
                    print(f"   Sample cache LADs: {sample_cache_lads}")
                    
                    # Test mapping for a specific LAD
                    test_lad = "Barnsley Borough Council"
                    mapped = self._map_lad_to_standard(test_lad)
                    normalized = self._normalize_lad_name(test_lad)
                    print(f"📰 Test mapping: '{test_lad}' -> normalized: '{normalized}' -> mapped: '{mapped}'")
                else:
                    print("📰 LAD mapping cache is None!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading local news data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_location_mapping(self):
        """Initialize location mapping for LADs and places"""
        try:
            # Load LAD mapping data
            lad_file = "data/Local_Authority_Districts_May_2023.csv"
            if os.path.exists(lad_file):
                lad_df = pd.read_csv(lad_file)
                self._lad_mapping_cache = lad_df
                print(f"📰 Loaded LAD mapping data: {len(lad_df)} LADs")
                print(f"📰 LAD mapping columns: {list(lad_df.columns)}")
            else:
                print("⚠️ LAD mapping file not found, using basic mapping")
                self._lad_mapping_cache = None
            
            # Initialize place mapping cache
            self._place_mapping_cache = {}
            
        except Exception as e:
            print(f"⚠️ Error initializing location mapping: {e}")
            self._lad_mapping_cache = None
            self._place_mapping_cache = {}
    
    def _normalize_lad_name(self, name: str) -> str:
        """Normalize LAD name for better matching"""
        if pd.isna(name):
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = str(name).lower().strip()
        
        # Remove common suffixes in order of specificity (longest first)
        suffixes_to_remove = [
            'metropolitan borough council',
            'borough council', 
            'city council', 
            'district council', 
            'unitary authority',
            'council', 
            'borough', 
            'district', 
            'city'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break
        
        return normalized
    
    def _map_lad_to_standard(self, lad_name: str) -> Optional[str]:
        """Map a LAD name to standard LAD format"""
        if not lad_name or pd.isna(lad_name):
            return None
        
        normalized_input = self._normalize_lad_name(lad_name)
        
        if self._lad_mapping_cache is not None:
            # Try to find exact match in LAD mapping data
            for _, row in self._lad_mapping_cache.iterrows():
                # Try different possible column names for LAD name
                lad_standard = row.get('LAD24NM', '') or row.get('LAD23NM', '') or row.get('LAD22NM', '')
                if lad_standard and self._normalize_lad_name(lad_standard) == normalized_input:
                    return lad_standard
            
            # Try partial matching (input contains standard or vice versa)
            for _, row in self._lad_mapping_cache.iterrows():
                # Try different possible column names for LAD name
                lad_standard = row.get('LAD24NM', '') or row.get('LAD23NM', '') or row.get('LAD22NM', '')
                if lad_standard:
                    lad_standard_normalized = self._normalize_lad_name(lad_standard)
                    
                    # Check if normalized input is contained in standard name or vice versa
                    if (normalized_input in lad_standard_normalized and len(normalized_input) > 3) or \
                       (lad_standard_normalized in normalized_input and len(lad_standard_normalized) > 3):
                        return lad_standard
        
        # Fallback: return the original name if no mapping found
        return lad_name
    
    def _map_place_to_lad(self, place_name: str) -> Optional[str]:
        """Map a referenced place to a LAD"""
        if not place_name or pd.isna(place_name):
            return None
        
        # Check cache first
        if place_name in self._place_mapping_cache:
            return self._place_mapping_cache[place_name]
        
        normalized_place = self._normalize_lad_name(place_name)
        
        if self._lad_mapping_cache is not None:
            # Try to find the place in LAD names
            for _, row in self._lad_mapping_cache.iterrows():
                # Try different possible column names for LAD name
                lad_standard = row.get('LAD24NM', '') or row.get('LAD23NM', '') or row.get('LAD22NM', '')
                if lad_standard:
                    if normalized_place in self._normalize_lad_name(lad_standard) or \
                       self._normalize_lad_name(lad_standard) in normalized_place:
                        self._place_mapping_cache[place_name] = lad_standard
                        return lad_standard
        
        # Cache the result (even if None)
        self._place_mapping_cache[place_name] = None
        return None
    
    def get_news_data(self) -> Optional[pd.DataFrame]:
        """Get local news data with location mapping"""
        if not self._data_loaded:
            self.load_data()
        
        if self.news_data is None:
            return None
        
        # Create a copy with mapped locations
        mapped_data = self.news_data.copy()
        
        # Map local authority districts to standard LADs
        print("📰 Mapping local authority districts...")
        mapped_data['mapped_lad'] = mapped_data['local_authority_district'].apply(self._map_lad_to_standard)
        
        # Map referenced places to LADs
        print("📰 Mapping referenced places...")
        mapped_data['mapped_referenced_place'] = mapped_data['referenced_place'].apply(self._map_place_to_lad)
        
        # Debug: Show mapping results
        print("📰 Mapping results:")
        print(f"   Articles with mapped LADs: {mapped_data['mapped_lad'].notna().sum()}/{len(mapped_data)}")
        print(f"   Articles with mapped places: {mapped_data['mapped_referenced_place'].notna().sum()}/{len(mapped_data)}")
        print(f"   Unique mapped LADs: {mapped_data['mapped_lad'].dropna().nunique()}")
        print(f"   Unique mapped places: {mapped_data['mapped_referenced_place'].dropna().nunique()}")
        
        return mapped_data
    
    def get_news_by_lad(self, lad_name: str) -> Optional[pd.DataFrame]:
        """Get news data for a specific LAD"""
        if not self._data_loaded:
            self.load_data()
        
        if self.news_data is None:
            return None
        
        mapped_data = self.get_news_data()
        if mapped_data is None:
            return None
        
        # Try exact match first
        lad_news = mapped_data[
            (mapped_data['mapped_lad'] == lad_name) |
            (mapped_data['mapped_referenced_place'] == lad_name)
        ]
        
        # If no exact match, try partial matching
        if lad_news.empty:
            # Normalize the input LAD name
            normalized_input = self._normalize_lad_name(lad_name)
            
            # Try partial matching for mapped_lad
            for _, row in mapped_data.iterrows():
                mapped_lad = row.get('mapped_lad')
                if pd.notna(mapped_lad):
                    mapped_lad_normalized = self._normalize_lad_name(mapped_lad)
                    if (normalized_input in mapped_lad_normalized and len(normalized_input) > 3) or \
                       (mapped_lad_normalized in normalized_input and len(mapped_lad_normalized) > 3):
                        lad_news = mapped_data[
                            (mapped_data['mapped_lad'] == mapped_lad) |
                            (mapped_data['mapped_referenced_place'] == mapped_lad)
                        ]
                        break
            
            # Try partial matching for mapped_referenced_place
            if lad_news.empty:
                for _, row in mapped_data.iterrows():
                    mapped_place = row.get('mapped_referenced_place')
                    if pd.notna(mapped_place):
                        mapped_place_normalized = self._normalize_lad_name(mapped_place)
                        if (normalized_input in mapped_place_normalized and len(normalized_input) > 3) or \
                           (mapped_place_normalized in normalized_input and len(mapped_place_normalized) > 3):
                            lad_news = mapped_data[
                                (mapped_data['mapped_lad'] == mapped_place) |
                                (mapped_data['mapped_referenced_place'] == mapped_place)
                            ]
                            break
        
        return lad_news if not lad_news.empty else None
    
    def get_news_summary(self) -> Dict[str, Any]:
        """Get summary of local news data"""
        if not self._data_loaded:
            self.load_data()
        
        if self.news_data is None:
            return {}
        
        mapped_data = self.get_news_data()
        if mapped_data is None:
            return {}
        
        total_articles = len(mapped_data)
        unique_lads = mapped_data['mapped_lad'].nunique()
        unique_places = mapped_data['mapped_referenced_place'].nunique()
        
        # Count articles by LAD
        lad_counts = mapped_data['mapped_lad'].value_counts().head(10)
        
        # Count articles by source
        source_counts = mapped_data['source_id'].value_counts().head(10)
        
        return {
            'total_articles': total_articles,
            'unique_lads_covered': unique_lads,
            'unique_places_referenced': unique_places,
            'top_lads_by_articles': lad_counts.to_dict(),
            'top_sources': source_counts.to_dict(),
            'mapping_success_rate': {
                'lad_mapping': (mapped_data['mapped_lad'].notna().sum() / total_articles) * 100,
                'place_mapping': (mapped_data['mapped_referenced_place'].notna().sum() / total_articles) * 100
            }
        }
    
    def get_all_lad_names(self) -> List[str]:
        """Get all unique LAD names from the news data"""
        if self.news_data is None:
            return []
        
        mapped_data = self.get_news_data()
        if mapped_data is None:
            return []
        
        # Get unique LADs from both originating and referenced places
        originating_lads = mapped_data['mapped_lad'].dropna().unique()
        referenced_lads = mapped_data['mapped_referenced_place'].dropna().unique()
        
        all_lads = list(set(list(originating_lads) + list(referenced_lads)))
        return sorted(all_lads)
    
    def get_news_keywords_analysis(self) -> Dict[str, Any]:
        """Analyze keywords and themes in news articles"""
        if not self._data_loaded:
            self.load_data()
        
        if self.news_data is None:
            return {}
        
        # Common keywords related to social cohesion
        cohesion_keywords = [
            'hate crime', 'community', 'cohesion', 'integration', 'diversity',
            'inclusion', 'tolerance', 'respect', 'harmony', 'solidarity',
            'engagement', 'participation', 'volunteer', 'neighbourhood',
            'safety', 'security', 'trust', 'confidence', 'support'
        ]
        
        keyword_counts = {}
        for keyword in cohesion_keywords:
            count = self.news_data['brief_description'].str.lower().str.contains(keyword, na=False).sum()
            if count > 0:
                keyword_counts[keyword] = count
        
        return {
            'cohesion_keywords': keyword_counts,
            'total_keyword_matches': sum(keyword_counts.values()),
            'articles_with_keywords': len(self.news_data[
                self.news_data['brief_description'].str.lower().str.contains('|'.join(cohesion_keywords), na=False)
            ])
        }
