#!/usr/bin/env python3
"""
News Sources Finder
Simple tool to search for local news sources by area name
"""

import json
import os
from typing import List, Dict, Optional

class NewsSourcesFinder:
    def __init__(self, data_file: str = "Data/news_gb.geojson"):
        """Initialize with path to news sources data file"""
        self.data_file = data_file
        self.sources = []
        self.load_sources()

    def load_sources(self):
        """Load news sources from the JSON/GeoJSON file"""
        if not os.path.exists(self.data_file):
            print(f"❌ Data file not found: {self.data_file}")
            print("Creating sample data file...")
            self.create_sample_data()
            return

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle GeoJSON format
            if 'features' in data:
                for feature in data['features']:
                    if 'properties' in feature:
                        props = feature['properties']
                        coords = feature.get('geometry', {}).get('coordinates', [0, 0])

                        source = {
                            'name': props.get('name', ''),
                            'town': props.get('town', ''),
                            'address': props.get('address', ''),
                            'postcode': props.get('postcode', ''),
                            'country': props.get('country', ''),
                            'url': props.get('url', ''),
                            'lat': float(props.get('lat', coords[1] if len(coords) > 1 else 0)),
                            'long': float(props.get('long', coords[0] if len(coords) > 0 else 0))
                        }
                        self.sources.append(source)
            else:
                # Handle plain JSON array
                self.sources = data

            print(f"✅ Loaded {len(self.sources)} news sources")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Creating sample data file...")
            self.create_sample_data()

    def create_sample_data(self):
        """Create sample news sources data for demonstration"""
        sample_sources = [
            {
                "name": "Brighton and Hove Independent",
                "town": "Brighton",
                "address": "34-36 St Leonards Road",
                "postcode": "BN21 3UT",
                "country": "England",
                "url": "https://www.brightonandhoveindependent.co.uk",
                "lat": 50.823941,
                "long": -0.169466
            },
            {
                "name": "Manchester Evening News",
                "town": "Manchester",
                "address": "1 Scott Place",
                "postcode": "M3 3RN",
                "country": "England",
                "url": "https://www.manchestereveningnews.co.uk",
                "lat": 53.4808,
                "long": -2.2426
            },
            {
                "name": "Birmingham Live",
                "town": "Birmingham",
                "address": "Weaman Street",
                "postcode": "B4 6AT",
                "country": "England",
                "url": "https://www.birminghamlive.co.uk",
                "lat": 52.4862,
                "long": -1.8904
            },
            {
                "name": "Liverpool Echo",
                "town": "Liverpool",
                "address": "Old Hall Street",
                "postcode": "L69 3EB",
                "country": "England",
                "url": "https://www.liverpoolecho.co.uk",
                "lat": 53.4084,
                "long": -2.9916
            },
            {
                "name": "Yorkshire Evening Post",
                "town": "Leeds",
                "address": "No 1 Leeds",
                "postcode": "LS2 7EE",
                "country": "England",
                "url": "https://www.yorkshireeveningpost.co.uk",
                "lat": 53.8008,
                "long": -1.5491
            },
            {
                "name": "The Herald",
                "town": "Glasgow",
                "address": "200 Renfield Street",
                "postcode": "G2 3QB",
                "country": "Scotland",
                "url": "https://www.heraldscotland.com",
                "lat": 55.8642,
                "long": -4.2518
            },
            {
                "name": "Edinburgh Evening News",
                "town": "Edinburgh",
                "address": "Orchard Brae House",
                "postcode": "EH4 2HS",
                "country": "Scotland",
                "url": "https://www.edinburghnews.scotsman.com",
                "lat": 55.9533,
                "long": -3.1883
            },
            {
                "name": "Wales Online",
                "town": "Cardiff",
                "address": "6 Park Street",
                "postcode": "CF10 1XR",
                "country": "Wales",
                "url": "https://www.walesonline.co.uk",
                "lat": 51.4816,
                "long": -3.1791
            },
            {
                "name": "Bristol Post",
                "town": "Bristol",
                "address": "Temple Way",
                "postcode": "BS99 7HD",
                "country": "England",
                "url": "https://www.bristolpost.co.uk",
                "lat": 51.4545,
                "long": -2.5879
            },
            {
                "name": "Chronicle Live",
                "town": "Newcastle",
                "address": "Groat Market",
                "postcode": "NE1 1ED",
                "country": "England",
                "url": "https://www.chroniclelive.co.uk",
                "lat": 54.9783,
                "long": -1.6178
            }
        ]

        # Create Data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)

        # Save as GeoJSON format
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }

        for source in sample_sources:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [source['long'], source['lat']]
                },
                "properties": source
            }
            geojson_data["features"].append(feature)

        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)

        self.sources = sample_sources
        print(f"✅ Created sample data file with {len(self.sources)} sources")

    def search_by_area(self, area_name: str) -> List[Dict]:
        """Search for news sources by area name (city, town, region)"""
        area_lower = area_name.lower().strip()
        matches = []

        for source in self.sources:
            # Search in town, address, and source name
            if (area_lower in source.get('town', '').lower() or
                area_lower in source.get('address', '').lower() or
                area_lower in source.get('name', '').lower() or
                area_lower in source.get('country', '').lower() or
                area_lower in source.get('postcode', '').lower()):
                matches.append(source)

        return matches

    def search_nearby(self, lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
        """Search for news sources within a radius of coordinates"""
        from math import radians, cos, sin, asin, sqrt

        def haversine(lon1, lat1, lon2, lat2):
            """Calculate distance between two points on Earth"""
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Earth's radius in km
            return c * r

        nearby = []
        for source in self.sources:
            distance = haversine(lon, lat, source.get('long', 0), source.get('lat', 0))
            if distance <= radius_km:
                source_with_distance = source.copy()
                source_with_distance['distance_km'] = round(distance, 2)
                nearby.append(source_with_distance)

        # Sort by distance
        nearby.sort(key=lambda x: x['distance_km'])
        return nearby

    def print_sources(self, sources: List[Dict], title: str = "News Sources"):
        """Print sources in a formatted way"""
        if not sources:
            print(f"\n❌ No {title.lower()} found")
            return

        print(f"\n📰 {title} ({len(sources)} found):")
        print("=" * 60)

        for i, source in enumerate(sources, 1):
            print(f"\n{i}. {source.get('name', 'Unknown')}")
            print(f"   🏢 Location: {source.get('town', 'Unknown')}, {source.get('country', 'Unknown')}")
            if source.get('address'):
                print(f"   📍 Address: {source.get('address')}")
            if source.get('postcode'):
                print(f"   📮 Postcode: {source.get('postcode')}")
            print(f"   🌐 Website: {source.get('url', 'N/A')}")
            print(f"   📍 Coordinates: ({source.get('lat', 0):.4f}, {source.get('long', 0):.4f})")
            if 'distance_km' in source:
                print(f"   📏 Distance: {source['distance_km']} km")

    def list_all_areas(self) -> List[str]:
        """Get a list of all unique areas (towns/cities)"""
        areas = set()
        for source in self.sources:
            if source.get('town'):
                areas.add(source['town'])
        return sorted(list(areas))

    def get_statistics(self) -> Dict:
        """Get statistics about the news sources"""
        countries = {}
        towns = {}

        for source in self.sources:
            country = source.get('country', 'Unknown')
            town = source.get('town', 'Unknown')

            countries[country] = countries.get(country, 0) + 1
            towns[town] = towns.get(town, 0) + 1

        return {
            'total_sources': len(self.sources),
            'countries': dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)),
            'towns': dict(sorted(towns.items(), key=lambda x: x[1], reverse=True))
        }

def main():
    """Main function to run the interactive news sources finder"""
    print("📰 News Sources Finder")
    print("=" * 40)

    # Initialize finder
    finder = NewsSourcesFinder()

    if not finder.sources:
        print("❌ No sources loaded. Exiting.")
        return

    while True:
        print("\n" + "=" * 40)
        print("Choose an option:")
        print("1. Search by area name")
        print("2. Search by coordinates")
        print("3. List all available areas")
        print("4. Show statistics")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            area = input("\n🔍 Enter area name (city, town, region): ").strip()
            if area:
                sources = finder.search_by_area(area)
                finder.print_sources(sources, f"Sources matching '{area}'")
            else:
                print("❌ Please enter an area name")

        elif choice == '2':
            try:
                lat = float(input("\n📍 Enter latitude: "))
                lon = float(input("📍 Enter longitude: "))
                radius = float(input("📏 Enter search radius in km (default 50): ") or 50)

                sources = finder.search_nearby(lat, lon, radius)
                finder.print_sources(sources, f"Sources within {radius}km of ({lat}, {lon})")

            except ValueError:
                print("❌ Please enter valid numbers for coordinates")

        elif choice == '3':
            areas = finder.list_all_areas()
            print(f"\n📍 Available areas ({len(areas)}):")
            print("-" * 30)
            for i, area in enumerate(areas, 1):
                print(f"{i:2d}. {area}")

        elif choice == '4':
            stats = finder.get_statistics()
            print(f"\n📊 Statistics:")
            print(f"Total sources: {stats['total_sources']}")

            print(f"\n🏴 By Country:")
            for country, count in list(stats['countries'].items())[:5]:
                print(f"  {country}: {count}")

            print(f"\n🏢 Top Towns:")
            for town, count in list(stats['towns'].items())[:10]:
                print(f"  {town}: {count}")

        elif choice == '5':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()