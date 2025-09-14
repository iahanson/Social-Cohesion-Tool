"""
Locality to MSOA Mapping System
Comprehensive mapping of localities, postcodes, and landmarks to MSOA codes
"""

import os
import json
import re
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

@dataclass
class LocalityInfo:
    """Information about a locality"""
    name: str
    type: str  # postcode, borough, ward, landmark, area
    msoa_code: str
    local_authority: str
    region: str
    coordinates: Optional[Tuple[float, float]] = None
    confidence: float = 1.0

class LocalityMapper:
    """Maps localities to MSOA codes and handles MSOA operations"""
    
    def __init__(self):
        self.msoa_data = self._load_msoa_data()
        self.postcode_mapping = self._load_postcode_mapping()
        self.landmark_mapping = self._load_landmark_mapping()
        self.area_mapping = self._load_area_mapping()
        
        # API endpoints for live data
        self.ons_postcode_url = "https://api.postcodes.io/postcodes/"
        self.msoa_boundaries_url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/MSOA_Names_and_Codes_England_and_Wales/FeatureServer/0/query"
    
    def _load_msoa_data(self) -> Dict[str, Dict[str, str]]:
        """Load MSOA data with local authority mappings"""
        return {
            "E02000001": {"la": "Kensington and Chelsea", "region": "London"},
            "E02000002": {"la": "Westminster", "region": "London"},
            "E02000003": {"la": "Hammersmith and Fulham", "region": "London"},
            "E02000004": {"la": "Wandsworth", "region": "London"},
            "E02000005": {"la": "Lambeth", "region": "London"},
            "E02000006": {"la": "Southwark", "region": "London"},
            "E02000007": {"la": "Tower Hamlets", "region": "London"},
            "E02000008": {"la": "Hackney", "region": "London"},
            "E02000009": {"la": "Islington", "region": "London"},
            "E02000010": {"la": "Camden", "region": "London"},
            "E02000011": {"la": "Greenwich", "region": "London"},
            "E02000012": {"la": "Lewisham", "region": "London"},
            "E02000013": {"la": "Southwark", "region": "London"},
            "E02000014": {"la": "Bromley", "region": "London"},
            "E02000015": {"la": "Croydon", "region": "London"},
            "E02000016": {"la": "Merton", "region": "London"},
            "E02000017": {"la": "Sutton", "region": "London"},
            "E02000018": {"la": "Kingston upon Thames", "region": "London"},
            "E02000019": {"la": "Richmond upon Thames", "region": "London"},
            "E02000020": {"la": "Hounslow", "region": "London"},
            "E02000021": {"la": "Ealing", "region": "London"},
            "E02000022": {"la": "Brent", "region": "London"},
            "E02000023": {"la": "Harrow", "region": "London"},
            "E02000024": {"la": "Barnet", "region": "London"},
            "E02000025": {"la": "Enfield", "region": "London"},
            "E02000026": {"la": "Haringey", "region": "London"},
            "E02000027": {"la": "Waltham Forest", "region": "London"},
            "E02000028": {"la": "Redbridge", "region": "London"},
            "E02000029": {"la": "Newham", "region": "London"},
            "E02000030": {"la": "Barking and Dagenham", "region": "London"},
            "E02000031": {"la": "Havering", "region": "London"},
            "E02000032": {"la": "Bexley", "region": "London"}
        }
    
    def _load_postcode_mapping(self) -> Dict[str, str]:
        """Load postcode to MSOA mapping"""
        return {
            # Central London
            "SW1": "E02000001",  # Kensington and Chelsea
            "SW3": "E02000001",
            "SW5": "E02000001",
            "SW7": "E02000001",
            "SW10": "E02000001",
            "W1": "E02000002",   # Westminster
            "W2": "E02000002",
            "W8": "E02000002",
            "W9": "E02000002",
            "W10": "E02000002",
            "W11": "E02000002",
            "W12": "E02000002",
            "WC1": "E02000009",  # Islington
            "WC2": "E02000009",
            "EC1": "E02000008",   # Hackney
            "EC2": "E02000008",
            "EC3": "E02000007",   # Tower Hamlets
            "EC4": "E02000007",
            "SE1": "E02000006",   # Southwark
            "SE2": "E02000006",
            "SE3": "E02000011",   # Greenwich
            "SE4": "E02000012",   # Lewisham
            "SE5": "E02000005",   # Lambeth
            "SE6": "E02000012",
            "SE7": "E02000011",
            "SE8": "E02000012",
            "SE9": "E02000014",   # Bromley
            "SE10": "E02000011",
            "SE11": "E02000005",
            "SE12": "E02000012",
            "SE13": "E02000012",
            "SE14": "E02000012",
            "SE15": "E02000006",
            "SE16": "E02000006",
            "SE17": "E02000005",
            "SE18": "E02000011",
            "SE19": "E02000015",  # Croydon
            "SE20": "E02000015",
            "SE21": "E02000005",
            "SE22": "E02000006",
            "SE23": "E02000012",
            "SE24": "E02000005",
            "SE25": "E02000015",
            "SE26": "E02000012",
            "SE27": "E02000005",
            "SE28": "E02000011",
            "N1": "E02000009",    # Islington
            "N2": "E02000024",    # Barnet
            "N3": "E02000024",
            "N4": "E02000026",    # Haringey
            "N5": "E02000009",
            "N6": "E02000026",
            "N7": "E02000009",
            "N8": "E02000026",
            "N9": "E02000025",    # Enfield
            "N10": "E02000026",
            "N11": "E02000025",
            "N12": "E02000024",
            "N13": "E02000025",
            "N14": "E02000025",
            "N15": "E02000026",
            "N16": "E02000008",
            "N17": "E02000025",
            "N18": "E02000025",
            "N19": "E02000009",
            "N20": "E02000024",
            "N21": "E02000025",
            "N22": "E02000026",
            "E1": "E02000007",    # Tower Hamlets
            "E2": "E02000008",    # Hackney
            "E3": "E02000007",
            "E4": "E02000027",    # Waltham Forest
            "E5": "E02000027",
            "E6": "E02000029",    # Newham
            "E7": "E02000029",
            "E8": "E02000008",
            "E9": "E02000008",
            "E10": "E02000027",
            "E11": "E02000028",   # Redbridge
            "E12": "E02000029",
            "E13": "E02000030",   # Barking and Dagenham
            "E14": "E02000007",
            "E15": "E02000029",
            "E16": "E02000030",
            "E17": "E02000027",
            "E18": "E02000028",
            "E20": "E02000029",
            "NW1": "E02000010",   # Camden
            "NW2": "E02000010",
            "NW3": "E02000010",
            "NW4": "E02000024",
            "NW5": "E02000010",
            "NW6": "E02000010",
            "NW7": "E02000024",
            "NW8": "E02000010",
            "NW9": "E02000010",
            "NW10": "E02000020",  # Hounslow
            "NW11": "E02000024",
            "SW1": "E02000001",
            "SW2": "E02000005",   # Lambeth
            "SW4": "E02000005",
            "SW6": "E02000003",   # Hammersmith and Fulham
            "SW8": "E02000005",
            "SW9": "E02000005",
            "SW11": "E02000004",  # Wandsworth
            "SW12": "E02000004",
            "SW13": "E02000019",  # Richmond upon Thames
            "SW14": "E02000019",
            "SW15": "E02000019",
            "SW16": "E02000015",  # Croydon
            "SW17": "E02000016",  # Merton
            "SW18": "E02000004",
            "SW19": "E02000016",
            "SW20": "E02000016"
        }
    
    def _load_landmark_mapping(self) -> Dict[str, str]:
        """Load landmark to MSOA mapping"""
        return {
            # Major landmarks and areas
            "hyde park": "E02000002",
            "kensington gardens": "E02000001",
            "regent's park": "E02000010",
            "greenwich park": "E02000011",
            "battersea park": "E02000004",
            "clapham common": "E02000005",
            "hampstead heath": "E02000010",
            "richmond park": "E02000019",
            "buckingham palace": "E02000002",
            "westminster abbey": "E02000002",
            "big ben": "E02000002",
            "london eye": "E02000002",
            "tower bridge": "E02000007",
            "tower of london": "E02000007",
            "st paul's cathedral": "E02000008",
            "covent garden": "E02000009",
            "leicester square": "E02000002",
            "piccadilly circus": "E02000002",
            "oxford street": "E02000002",
            "regent street": "E02000002",
            "bond street": "E02000002",
            "carnaby street": "E02000002",
            "soho": "E02000002",
            "covent garden": "E02000009",
            "camden market": "E02000010",
            "portobello road": "E02000001",
            "notting hill": "E02000001",
            "chelsea": "E02000001",
            "south kensington": "E02000001",
            "knightsbridge": "E02000001",
            "belgravia": "E02000002",
            "mayfair": "E02000002",
            "marylebone": "E02000002",
            "fitzrovia": "E02000009",
            "bloomsbury": "E02000009",
            "king's cross": "E02000009",
            "angel": "E02000009",
            "islington": "E02000009",
            "camden town": "E02000010",
            "primrose hill": "E02000010",
            "hampstead": "E02000010",
            "belsize park": "E02000010",
            "swiss cottage": "E02000010",
            "st john's wood": "E02000010",
            "maida vale": "E02000010",
            "paddington": "E02000002",
            "bayswater": "E02000002",
            "marylebone": "E02000002",
            "fitzrovia": "E02000009",
            "holborn": "E02000009",
            "clerkenwell": "E02000008",
            "shoreditch": "E02000008",
            "spitalfields": "E02000007",
            "whitechapel": "E02000007",
            "canary wharf": "E02000007",
            "docklands": "E02000007",
            "greenwich": "E02000011",
            "deptford": "E02000012",
            "lewisham": "E02000012",
            "new cross": "E02000012",
            "peckham": "E02000006",
            "camberwell": "E02000005",
            "brixton": "E02000005",
            "clapham": "E02000005",
            "battersea": "E02000004",
            "wandsworth": "E02000004",
            "putney": "E02000004",
            "fulham": "E02000003",
            "hammersmith": "E02000003",
            "chiswick": "E02000003",
            "shepherd's bush": "E02000003",
            "white city": "E02000003",
            "acton": "E02000021",
            "ealing": "E02000021",
            "harrow": "E02000023",
            "wembley": "E02000022",
            "brent": "E02000022",
            "willesden": "E02000022",
            "kilburn": "E02000022",
            "maida vale": "E02000010",
            "st john's wood": "E02000010",
            "swiss cottage": "E02000010",
            "belsize park": "E02000010",
            "primrose hill": "E02000010",
            "camden town": "E02000010",
            "kentish town": "E02000010",
            "tufnell park": "E02000010",
            "archway": "E02000010",
            "highgate": "E02000010",
            "muswell hill": "E02000026",
            "crouch end": "E02000026",
            "hornsey": "E02000026",
            "wood green": "E02000026",
            "tottenham": "E02000025",
            "edmonton": "E02000025",
            "enfield": "E02000025",
            "winchmore hill": "E02000025",
            "southgate": "E02000025",
            "palmers green": "E02000025",
            "new southgate": "E02000025",
            "cockfosters": "E02000025",
            "barnet": "E02000024",
            "east barnet": "E02000024",
            "new barnet": "E02000024",
            "mill hill": "E02000024",
            "hendon": "E02000024",
            "colindale": "E02000024",
            "burnt oak": "E02000024",
            "edgware": "E02000024",
            "stanmore": "E02000024",
            "harrow on the hill": "E02000023",
            "harrow weald": "E02000023",
            "wealdstone": "E02000023",
            "kenton": "E02000023",
            "south kenton": "E02000023",
            "northwick park": "E02000023",
            "pinner": "E02000023",
            "north harrow": "E02000023",
            "west harrow": "E02000023",
            "headstone": "E02000023",
            "harrow": "E02000023",
            "wembley": "E02000022",
            "wembley park": "E02000022",
            "wembley central": "E02000022",
            "sudbury": "E02000022",
            "sudbury hill": "E02000022",
            "sudbury town": "E02000022",
            "alperton": "E02000022",
            "park royal": "E02000022",
            "stonebridge": "E02000022",
            "neasden": "E02000022",
            "dollis hill": "E02000022",
            "willesden": "E02000022",
            "willesden green": "E02000022",
            "kilburn": "E02000022",
            "queen's park": "E02000022",
            "brondesbury": "E02000022",
            "maida vale": "E02000010",
            "st john's wood": "E02000010",
            "swiss cottage": "E02000010",
            "belsize park": "E02000010",
            "primrose hill": "E02000010",
            "camden town": "E02000010",
            "kentish town": "E02000010",
            "tufnell park": "E02000010",
            "archway": "E02000010",
            "highgate": "E02000010",
            "muswell hill": "E02000026",
            "crouch end": "E02000026",
            "hornsey": "E02000026",
            "wood green": "E02000026",
            "tottenham": "E02000025",
            "edmonton": "E02000025",
            "enfield": "E02000025",
            "winchmore hill": "E02000025",
            "southgate": "E02000025",
            "palmers green": "E02000025",
            "new southgate": "E02000025",
            "cockfosters": "E02000025",
            "barnet": "E02000024",
            "east barnet": "E02000024",
            "new barnet": "E02000024",
            "mill hill": "E02000024",
            "hendon": "E02000024",
            "colindale": "E02000024",
            "burnt oak": "E02000024",
            "edgware": "E02000024",
            "stanmore": "E02000024",
            "harrow on the hill": "E02000023",
            "harrow weald": "E02000023",
            "wealdstone": "E02000023",
            "kenton": "E02000023",
            "south kenton": "E02000023",
            "northwick park": "E02000023",
            "pinner": "E02000023",
            "north harrow": "E02000023",
            "west harrow": "E02000023",
            "headstone": "E02000023",
            "harrow": "E02000023",
            "wembley": "E02000022",
            "wembley park": "E02000022",
            "wembley central": "E02000022",
            "sudbury": "E02000022",
            "sudbury hill": "E02000022",
            "sudbury town": "E02000022",
            "alperton": "E02000022",
            "park royal": "E02000022",
            "stonebridge": "E02000022",
            "neasden": "E02000022",
            "dollis hill": "E02000022",
            "willesden": "E02000022",
            "willesden green": "E02000022",
            "kilburn": "E02000022",
            "queen's park": "E02000022",
            "brondesbury": "E02000022"
        }
    
    def _load_area_mapping(self) -> Dict[str, str]:
        """Load area names to MSOA mapping"""
        return {
            # Borough names
            "kensington and chelsea": "E02000001",
            "royal borough of kensington and chelsea": "E02000001",
            "rbkc": "E02000001",
            "westminster": "E02000002",
            "city of westminster": "E02000002",
            "hammersmith and fulham": "E02000003",
            "lb hammersmith and fulham": "E02000003",
            "wandsworth": "E02000004",
            "lb wandsworth": "E02000004",
            "lambeth": "E02000005",
            "lb lambeth": "E02000005",
            "southwark": "E02000006",
            "lb southwark": "E02000006",
            "tower hamlets": "E02000007",
            "lb tower hamlets": "E02000007",
            "hackney": "E02000008",
            "lb hackney": "E02000008",
            "islington": "E02000009",
            "lb islington": "E02000009",
            "camden": "E02000010",
            "lb camden": "E02000010",
            "greenwich": "E02000011",
            "royal borough of greenwich": "E02000011",
            "rb greenwich": "E02000011",
            "lewisham": "E02000012",
            "lb lewisham": "E02000012",
            "bromley": "E02000014",
            "lb bromley": "E02000014",
            "croydon": "E02000015",
            "lb croydon": "E02000015",
            "merton": "E02000016",
            "lb merton": "E02000016",
            "sutton": "E02000017",
            "lb sutton": "E02000017",
            "kingston upon thames": "E02000018",
            "royal borough of kingston upon thames": "E02000018",
            "rb kingston": "E02000018",
            "richmond upon thames": "E02000019",
            "lb richmond upon thames": "E02000019",
            "hounslow": "E02000020",
            "lb hounslow": "E02000020",
            "ealing": "E02000021",
            "lb ealing": "E02000021",
            "brent": "E02000022",
            "lb brent": "E02000022",
            "harrow": "E02000023",
            "lb harrow": "E02000023",
            "barnet": "E02000024",
            "lb barnet": "E02000024",
            "enfield": "E02000025",
            "lb enfield": "E02000025",
            "haringey": "E02000026",
            "lb haringey": "E02000026",
            "waltham forest": "E02000027",
            "lb waltham forest": "E02000027",
            "redbridge": "E02000028",
            "lb redbridge": "E02000028",
            "newham": "E02000029",
            "lb newham": "E02000029",
            "barking and dagenham": "E02000030",
            "lb barking and dagenham": "E02000030",
            "havering": "E02000031",
            "lb havering": "E02000031",
            "bexley": "E02000032",
            "lb bexley": "E02000032"
        }
    
    def map_locality(self, locality: str) -> Optional[LocalityInfo]:
        """
        Map a locality to MSOA information
        
        Args:
            locality: Name of the locality to map
            
        Returns:
            LocalityInfo object or None if not found
        """
        if not locality:
            return None
        
        locality_clean = locality.lower().strip()
        
        # Try different mapping approaches
        msoa_code = None
        locality_type = "unknown"
        
        # 1. Try postcode mapping
        msoa_code = self._map_postcode(locality_clean)
        if msoa_code:
            locality_type = "postcode"
        
        # 2. Try landmark mapping
        if not msoa_code:
            msoa_code = self._map_landmark(locality_clean)
            if msoa_code:
                locality_type = "landmark"
        
        # 3. Try area mapping
        if not msoa_code:
            msoa_code = self._map_area(locality_clean)
            if msoa_code:
                locality_type = "area"
        
        # 4. Try fuzzy matching
        if not msoa_code:
            msoa_code, locality_type = self._fuzzy_match(locality_clean)
        
        if msoa_code:
            msoa_info = self.msoa_data.get(msoa_code, {})
            return LocalityInfo(
                name=locality,
                type=locality_type,
                msoa_code=msoa_code,
                local_authority=msoa_info.get("la", "Unknown"),
                region=msoa_info.get("region", "Unknown"),
                confidence=self._calculate_confidence(locality_clean, locality_type)
            )
        
        return None
    
    def _map_postcode(self, locality: str) -> Optional[str]:
        """Map postcode to MSOA"""
        # Extract postcode pattern
        postcode_match = re.match(r'^([a-z]{1,2}\d{1,2}[a-z]?)', locality)
        if postcode_match:
            postcode_prefix = postcode_match.group(1).upper()
            return self.postcode_mapping.get(postcode_prefix)
        
        # Try direct lookup
        return self.postcode_mapping.get(locality.upper())
    
    def _map_landmark(self, locality: str) -> Optional[str]:
        """Map landmark to MSOA"""
        return self.landmark_mapping.get(locality)
    
    def _map_area(self, locality: str) -> Optional[str]:
        """Map area name to MSOA"""
        return self.area_mapping.get(locality)
    
    def _fuzzy_match(self, locality: str) -> Tuple[Optional[str], str]:
        """Perform fuzzy matching for locality names"""
        # Simple fuzzy matching - look for partial matches
        for landmark, msoa_code in self.landmark_mapping.items():
            if locality in landmark or landmark in locality:
                return msoa_code, "landmark"
        
        for area, msoa_code in self.area_mapping.items():
            if locality in area or area in locality:
                return msoa_code, "area"
        
        return None, "unknown"
    
    def _calculate_confidence(self, locality: str, locality_type: str) -> float:
        """Calculate confidence score for the mapping"""
        if locality_type == "postcode":
            return 0.95  # High confidence for postcodes
        elif locality_type == "landmark":
            return 0.85  # Good confidence for landmarks
        elif locality_type == "area":
            return 0.90  # High confidence for area names
        else:
            return 0.50  # Lower confidence for fuzzy matches
    
    def map_multiple_localities(self, localities: List[str]) -> List[LocalityInfo]:
        """Map multiple localities to MSOA information"""
        results = []
        for locality in localities:
            mapped = self.map_locality(locality)
            if mapped:
                results.append(mapped)
        return results
    
    def get_msoa_info(self, msoa_code: str) -> Optional[Dict[str, str]]:
        """Get information about an MSOA code"""
        return self.msoa_data.get(msoa_code)
    
    def search_localities(self, query: str) -> List[LocalityInfo]:
        """Search for localities matching a query"""
        results = []
        query_lower = query.lower()
        
        # Search in landmarks
        for landmark, msoa_code in self.landmark_mapping.items():
            if query_lower in landmark:
                msoa_info = self.msoa_data.get(msoa_code, {})
                results.append(LocalityInfo(
                    name=landmark,
                    type="landmark",
                    msoa_code=msoa_code,
                    local_authority=msoa_info.get("la", "Unknown"),
                    region=msoa_info.get("region", "Unknown"),
                    confidence=0.85
                ))
        
        # Search in areas
        for area, msoa_code in self.area_mapping.items():
            if query_lower in area:
                msoa_info = self.msoa_data.get(msoa_code, {})
                results.append(LocalityInfo(
                    name=area,
                    type="area",
                    msoa_code=msoa_code,
                    local_authority=msoa_info.get("la", "Unknown"),
                    region=msoa_info.get("region", "Unknown"),
                    confidence=0.90
                ))
        
        return results
    
    def validate_msoa_code(self, msoa_code: str) -> bool:
        """Validate if an MSOA code exists"""
        return msoa_code in self.msoa_data
    
    def get_all_msoa_codes(self) -> List[str]:
        """Get all available MSOA codes"""
        return list(self.msoa_data.keys())
    
    def get_local_authorities(self) -> List[str]:
        """Get all local authorities"""
        return list(set(info["la"] for info in self.msoa_data.values()))
    
    def postcode_to_msoa(self, postcode: str) -> Optional[Dict[str, Any]]:
        """
        Convert UK postcode to MSOA information using postcodes.io API
        
        Args:
            postcode: UK postcode (with or without space)
            
        Returns:
            Dictionary containing MSOA information or None if not found
        """
        try:
            # Clean postcode
            clean_postcode = postcode.replace(" ", "").upper()
            
            # Use postcodes.io API
            response = requests.get(f"{self.ons_postcode_url}{clean_postcode}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 200:
                    result = data['result']
                    return {
                        'postcode': result['postcode'],
                        'msoa_code': result.get('codes', {}).get('msoa'),
                        'msoa_name': result.get('codes', {}).get('msoa'),
                        'lsoa_code': result.get('codes', {}).get('lsoa'),
                        'ward_code': result.get('codes', {}).get('ward'),
                        'local_authority': result.get('admin_district'),
                        'region': result.get('region'),
                        'country': result.get('country'),
                        'longitude': result.get('longitude'),
                        'latitude': result.get('latitude')
                    }
            return None
            
        except Exception as e:
            print(f"Error looking up postcode {postcode}: {e}")
            return None
    
    def get_msoa_details(self, msoa_code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an MSOA by its code
        
        Args:
            msoa_code: MSOA code
            
        Returns:
            Dictionary containing MSOA details or None if not found
        """
        try:
            # Check if we have static data for this MSOA
            if msoa_code in self.msoa_data:
                msoa_info = self.msoa_data[msoa_code]
                return {
                    'msoa_code': msoa_code,
                    'msoa_name': f"MSOA {msoa_code}",
                    'local_authority': msoa_info['la'],
                    'region': msoa_info['region'],
                    'country': 'England',
                    'status': 'Valid MSOA code'
                }
            
            # If not in static data but valid format, return basic info
            if self.validate_msoa_code(msoa_code):
                return {
                    'msoa_code': msoa_code,
                    'msoa_name': f"MSOA {msoa_code}",
                    'country': 'England',
                    'status': 'Valid MSOA code'
                }
            return None
        except Exception as e:
            print(f"Error getting MSOA details for {msoa_code}: {e}")
            return None
    
    def export_mapping_data(self, format: str = "json") -> str:
        """Export mapping data"""
        if format == "json":
            return json.dumps({
                "msoa_data": self.msoa_data,
                "postcode_mapping": self.postcode_mapping,
                "landmark_mapping": self.landmark_mapping,
                "area_mapping": self.area_mapping
            }, indent=2)
        elif format == "csv":
            # Export as CSV with MSOA codes and local authorities
            rows = []
            for msoa_code, info in self.msoa_data.items():
                rows.append({
                    "msoa_code": msoa_code,
                    "local_authority": info["la"],
                    "region": info["region"]
                })
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
