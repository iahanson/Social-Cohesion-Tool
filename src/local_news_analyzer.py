"""
Local News Analyzer for Social Cohesion Tool
Integrated news analysis with interactive mapping
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import folium
from collections import Counter
import streamlit as st
from geopy.distance import geodesic
from .traffic_analyzer import ArticleTrafficAnalyzer
from .news_agent_analyzer import NewsAgentAnalyzer

class LocalNewsAnalyzer:
    """Simplified news analyzer for Social Cohesion Tool integration"""

    def __init__(self):
        self.news_sources = self._load_uk_news_sources()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.traffic_analyzer = ArticleTrafficAnalyzer()
        self.agent_analyzer = NewsAgentAnalyzer()

        # Trust and community keywords
        self.trust_keywords = {
            'positive': ['community', 'together', 'support', 'help', 'volunteer', 'charity',
                        'cooperation', 'unity', 'celebrate', 'festival', 'success', 'achievement'],
            'negative': ['crime', 'violence', 'conflict', 'concern', 'problem', 'issue',
                        'crisis', 'struggle', 'protest', 'anger', 'fear', 'worry']
        }

        self.topic_keywords = {
            'housing': ['housing', 'homes', 'rent', 'property', 'development', 'planning'],
            'transport': ['transport', 'bus', 'train', 'traffic', 'road', 'parking'],
            'education': ['school', 'education', 'university', 'students', 'teachers'],
            'healthcare': ['hospital', 'health', 'NHS', 'medical', 'doctor', 'treatment'],
            'crime_safety': ['police', 'crime', 'safety', 'theft', 'antisocial', 'security'],
            'economy': ['jobs', 'employment', 'business', 'economy', 'shop', 'market'],
            'environment': ['environment', 'green', 'pollution', 'climate', 'park'],
            'local_gov': ['council', 'mayor', 'election', 'policy', 'budget']
        }

    def _load_uk_news_sources(self) -> List[Dict]:
        """Load UK news sources with coordinates"""
        return [
            {
                "name": "Manchester Evening News",
                "url": "https://www.manchestereveningnews.co.uk",
                "lat": 53.4808,
                "lon": -2.2426,
                "coverage_radius": 50
            },
            {
                "name": "Birmingham Live",
                "url": "https://www.birminghamlive.co.uk",
                "lat": 52.4862,
                "lon": -1.8904,
                "coverage_radius": 40
            },
            {
                "name": "Liverpool Echo",
                "url": "https://www.liverpoolecho.co.uk",
                "lat": 53.4084,
                "lon": -2.9916,
                "coverage_radius": 45
            },
            {
                "name": "Yorkshire Evening Post",
                "url": "https://www.yorkshireeveningpost.co.uk",
                "lat": 53.8008,
                "lon": -1.5491,
                "coverage_radius": 35
            },
            {
                "name": "Bristol Post",
                "url": "https://www.bristolpost.co.uk",
                "lat": 51.4545,
                "lon": -2.5879,
                "coverage_radius": 30
            },
            {
                "name": "Chronicle Live",
                "url": "https://www.chroniclelive.co.uk",
                "lat": 54.9783,
                "lon": -1.6178,
                "coverage_radius": 40
            },
            {
                "name": "The Herald (Glasgow)",
                "url": "https://www.heraldscotland.com",
                "lat": 55.8642,
                "lon": -4.2518,
                "coverage_radius": 60
            },
            {
                "name": "Wales Online",
                "url": "https://www.walesonline.co.uk",
                "lat": 51.4816,
                "lon": -3.1791,
                "coverage_radius": 50
            }
        ]

    def find_sources_for_location(self, lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
        """Find news sources covering a specific location"""
        location_sources = []

        for source in self.news_sources:
            # Calculate distance from source to location
            source_coords = (source['lat'], source['lon'])
            target_coords = (lat, lon)
            distance = geodesic(source_coords, target_coords).kilometers

            # Check if location is within source's coverage area
            if distance <= source['coverage_radius']:
                source_with_distance = source.copy()
                source_with_distance['distance_km'] = round(distance, 1)
                location_sources.append(source_with_distance)

        # Sort by distance
        location_sources.sort(key=lambda x: x['distance_km'])
        return location_sources

    def scrape_multiple_articles(self, source: Dict, max_articles: int = 5) -> List[Dict]:
        """Scrape multiple recent articles from a news source"""
        articles = []
        try:
            response = self.session.get(source['url'], timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find multiple articles
            articles_found = self._find_multiple_articles(soup, source['url'], max_articles)

            for article_data in articles_found:
                articles.append({
                    'title': article_data['title'],
                    'content': article_data['content'],
                    'url': article_data['url'],
                    'source': source['name'],
                    'scraped_at': datetime.now().isoformat(),
                    'popularity_score': article_data.get('popularity_score', 0)
                })

        except Exception as e:
            print(f"Error scraping {source['name']}: {e}")

        return articles

    def scrape_top_article(self, source: Dict) -> Optional[Dict]:
        """Scrape the most viewed/recent article from a news source (backward compatibility)"""
        # Use the new multiple articles method for backward compatibility
        articles = self.scrape_multiple_articles(source, max_articles=1)
        return articles[0] if articles else None

    def _find_main_article(self, soup: BeautifulSoup, base_url: str) -> Optional[Dict]:
        """Find the main/featured article on a news homepage"""
        # Look for main article selectors - expanded list for better coverage
        main_selectors = [
            '.lead-story', '.main-story', '.featured-story', '.hero-story', '.top-story',
            '.primary-story', '.headline', '.breaking-news', '.most-read', '.trending',
            'article:first-of-type', '.story-main', '.news-main', '.front-story',
            '[data-testid="story-headline"]', '.story-item:first-child',
            'h1 a', 'h2 a', '.title a', '.news-title a'
        ]

        for selector in main_selectors:
            elements = soup.select(selector)  # Get multiple matches
            for element in elements[:3]:  # Try first 3 matches
                try:
                    # Extract title - try multiple approaches
                    title = ''
                    if element.name == 'a':  # If selector found a link directly
                        title = element.get_text().strip()
                        article_url = element.get('href')
                    else:
                        # Look for title within element
                        title_elem = element.select_one('h1, h2, h3, h4, .headline, .title')
                        if title_elem:
                            title = title_elem.get_text().strip()

                        # Look for link within element
                        link_elem = element.select_one('a[href]')
                        if not link_elem:
                            continue
                        article_url = link_elem.get('href')

                    if not title or not article_url:
                        continue

                    # Make URL absolute
                    if article_url.startswith('/'):
                        article_url = base_url + article_url
                    elif not article_url.startswith('http'):
                        continue  # Skip invalid URLs

                    # Scrape full article content
                    content = self._scrape_article_content(article_url)

                    if title and content and len(content) > 100:  # Ensure we got real content
                        return {
                            'title': title,
                            'content': content,
                            'url': article_url
                        }
                except Exception as e:
                    print(f"Error processing article element: {e}")
                    continue

        # Fallback: try to find any recent article links
        links = soup.find_all('a', href=True)
        for link in links[:10]:  # Check first 10 links
            href = link.get('href')
            if href and ('article' in href or 'news' in href or '202' in href):  # Likely article URLs
                title = link.get_text().strip()
                if len(title) > 10:  # Reasonable title length
                    if href.startswith('/'):
                        href = base_url + href
                    content = self._scrape_article_content(href)
                    if content and len(content) > 100:
                        return {'title': title, 'content': content, 'url': href}

        return None

    def _scrape_article_content(self, url: str) -> str:
        """Scrape full content of an article"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Extract content
            content_selectors = [
                '.article-content', '.entry-content', '.post-content',
                'article', '.content', '.story-body'
            ]

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text()
                    content = ' '.join(content.split())  # Clean whitespace
                    if len(content) > 200:
                        return content

            # Fallback: get all paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return ' '.join(content.split())

        except Exception as e:
            print(f"Error scraping article content: {e}")
            return ""

    def analyze_article(self, article: Dict) -> Dict:
        """Analyze article for sentiment and topics"""
        text = f"{article['title']} {article['content']}".lower()
        words = text.split()

        # Sentiment analysis
        positive_count = sum(1 for word in words if word in self.trust_keywords['positive'])
        negative_count = sum(1 for word in words if word in self.trust_keywords['negative'])

        # Calculate sentiment score (-1 to 1)
        if positive_count + negative_count > 0:
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment = 0.0

        # Trust score (1 to 10)
        if positive_count + negative_count > 0:
            trust_score = (positive_count / (positive_count + negative_count)) * 10
        else:
            trust_score = 5.0

        # Topic analysis
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            topic_scores[topic] = count

        # Get top 3 topics
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_topics = [topic for topic, count in top_topics if count > 0]

        return {
            'sentiment_score': round(sentiment, 3),
            'trust_score': round(trust_score, 1),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'top_topics': top_topics,
            'topic_scores': topic_scores,
            'word_count': len(words)
        }

    def analyze_location(self, lat: float, lon: float, area_name: str = "") -> Dict:
        """Complete analysis for a clicked location"""
        # Find relevant news sources
        sources = self.find_sources_for_location(lat, lon)

        if not sources:
            return {
                'area_name': area_name,
                'coordinates': (lat, lon),
                'error': 'No news sources found for this location'
            }

        # Get the closest/most relevant source
        primary_source = sources[0]

        # Scrape top article
        article = self.scrape_top_article(primary_source)

        if not article:
            return {
                'area_name': area_name,
                'coordinates': (lat, lon),
                'source': primary_source['name'],
                'error': 'Could not retrieve article from news source'
            }

        # Analyze article
        analysis = self.analyze_article(article)

        # Now use comprehensive analysis instead
        return self.analyze_location_comprehensive(lat, lon, area_name)

    def analyze_location_comprehensive(self, lat: float, lon: float, area_name: str = "") -> Dict:
        """Comprehensive analysis for a clicked location - multiple sources and articles"""
        # Find relevant news sources
        sources = self.find_sources_for_location(lat, lon)

        if not sources:
            return {
                'area_name': area_name,
                'coordinates': (lat, lon),
                'error': 'No news sources found for this location'
            }

        all_articles = []
        source_summaries = []

        # Scrape from all available sources
        for source in sources[:3]:  # Limit to top 3 sources to avoid overload
            articles = self.scrape_multiple_articles(source, max_articles=3)
            if articles:
                all_articles.extend(articles)
                source_summaries.append({
                    'name': source['name'],
                    'distance': source['distance_km'],
                    'articles_found': len(articles)
                })

        if not all_articles:
            return {
                'area_name': area_name,
                'coordinates': (lat, lon),
                'sources': [s['name'] for s in sources],
                'error': 'Could not retrieve articles from any news source'
            }

        # Analyze traffic for all articles and get the most popular
        traffic_analyzed_articles = self.traffic_analyzer.analyze_article_traffic(all_articles)
        most_popular_article = traffic_analyzed_articles[0]

        # Analyze all articles for aggregate sentiment
        all_analyses = [self.analyze_article(article) for article in all_articles]

        # Calculate aggregate analysis
        aggregate_analysis = self._calculate_aggregate_analysis(all_analyses)

        # Detailed analysis of most popular article
        detailed_analysis = self.analyze_article(most_popular_article)

        # Get featured article metadata including image
        featured_article_metadata = self.traffic_analyzer.get_article_metadata(most_popular_article['url'])

        # Run AI agent analysis on all articles
        agent_analysis = self.agent_analyzer.analyze_articles(traffic_analyzed_articles)

        return {
            'area_name': area_name,
            'coordinates': (lat, lon),
            'sources_analyzed': source_summaries,
            'total_articles': len(traffic_analyzed_articles),
            'most_popular_article': {
                'title': most_popular_article['title'],
                'url': most_popular_article['url'],
                'source': most_popular_article['source'],
                'traffic_score': most_popular_article.get('traffic_score', 0),
                'traffic_indicators': most_popular_article.get('traffic_indicators', {}),
                'content_preview': most_popular_article['content'][:300] + '...' if len(most_popular_article['content']) > 300 else most_popular_article['content'],
                'image_url': featured_article_metadata.get('image', ''),
                'description': featured_article_metadata.get('description', ''),
                'site_name': featured_article_metadata.get('site_name', most_popular_article['source'])
            },
            'detailed_analysis': detailed_analysis,
            'aggregate_analysis': aggregate_analysis,
            'agent_analysis': agent_analysis,
            'all_articles': [{
                'title': art['title'],
                'source': art['source'],
                'url': art['url'],
                'traffic_score': art.get('traffic_score', 0),
                'traffic_indicators': art.get('traffic_indicators', {})
            } for art in traffic_analyzed_articles],
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _calculate_aggregate_analysis(self, analyses: List[Dict]) -> Dict:
        """Calculate aggregate sentiment and topics from multiple articles"""
        if not analyses:
            return {}

        # Average sentiment
        avg_sentiment = sum(a['sentiment_score'] for a in analyses) / len(analyses)

        # Average trust score
        avg_trust = sum(a['trust_score'] for a in analyses) / len(analyses)

        # Total indicators
        total_positive = sum(a['positive_indicators'] for a in analyses)
        total_negative = sum(a['negative_indicators'] for a in analyses)

        # Aggregate topic scores
        topic_totals = {}
        for analysis in analyses:
            for topic, score in analysis['topic_scores'].items():
                topic_totals[topic] = topic_totals.get(topic, 0) + score

        # Top aggregate topics
        top_topics = sorted(topic_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        top_topics = [topic for topic, count in top_topics if count > 0]

        return {
            'sentiment_score': round(avg_sentiment, 3),
            'trust_score': round(avg_trust, 1),
            'positive_indicators': total_positive,
            'negative_indicators': total_negative,
            'top_topics': top_topics,
            'topic_scores': topic_totals,
            'articles_analyzed': len(analyses),
            'confidence': 'High' if len(analyses) >= 3 else 'Medium' if len(analyses) >= 2 else 'Low'
        }

    def create_coverage_map(self) -> folium.Map:
        """Create a map showing news source coverage areas"""
        # Center map on UK
        uk_center = [54.7023, -3.2765]
        m = folium.Map(location=uk_center, zoom_start=6)

        for source in self.news_sources:
            # Add marker for news source
            folium.Marker(
                location=[source['lat'], source['lon']],
                popup=f"<b>{source['name']}</b><br>Coverage: {source['coverage_radius']}km",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

            # Add coverage circle
            folium.Circle(
                location=[source['lat'], source['lon']],
                radius=source['coverage_radius'] * 1000,  # Convert to meters
                popup=f"{source['name']} coverage area",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.2
            ).add_to(m)

        return m

    def create_interactive_map(self) -> str:
        """Create interactive map HTML for Streamlit"""
        m = self.create_coverage_map()

        # Add JavaScript for click handling
        click_js = """
        <script>
        function onMapClick(e) {
            var lat = e.latlng.lat;
            var lng = e.latlng.lng;

            // Send coordinates to Streamlit
            window.parent.postMessage({
                type: 'mapClick',
                lat: lat,
                lng: lng
            }, '*');
        }

        // Add click listener when map loads
        map.on('click', onMapClick);
        </script>
        """

        # Get map HTML
        map_html = m._repr_html_()

        # Inject click handler
        map_html = map_html.replace('</body>', click_js + '</body>')

        return map_html

    def _find_multiple_articles(self, soup: BeautifulSoup, base_url: str, max_articles: int = 5) -> List[Dict]:
        """Find multiple articles from a news homepage"""
        articles = []
        found_urls = set()  # Avoid duplicates

        # Priority selectors for main/featured articles
        priority_selectors = [
            '.lead-story', '.main-story', '.featured-story', '.hero-story', '.top-story',
            '.breaking-news', '.most-read', '.trending', '.popular'
        ]

        # General article selectors
        article_selectors = [
            'article', '.story-item', '.news-item', '.article-item', '.post-item',
            '.story', '.news-story', '[data-testid*="story"]', '.entry'
        ]

        # First, try to find priority/featured articles
        for selector in priority_selectors:
            elements = soup.select(selector)
            for element in elements[:2]:  # Max 2 priority articles
                article = self._extract_article_from_element(element, base_url)
                if article and article['url'] not in found_urls and len(articles) < max_articles:
                    article['popularity_score'] = 10  # High score for featured articles
                    articles.append(article)
                    found_urls.add(article['url'])

        # Then find additional articles to fill quota
        if len(articles) < max_articles:
            for selector in article_selectors:
                elements = soup.select(selector)
                for element in elements:
                    if len(articles) >= max_articles:
                        break

                    article = self._extract_article_from_element(element, base_url)
                    if article and article['url'] not in found_urls:
                        # Try to determine popularity based on position and indicators
                        article['popularity_score'] = self._calculate_popularity_score(element)
                        articles.append(article)
                        found_urls.add(article['url'])

        # Sort by popularity score
        articles.sort(key=lambda x: x['popularity_score'], reverse=True)
        return articles

    def _extract_article_from_element(self, element: BeautifulSoup, base_url: str) -> Optional[Dict]:
        """Extract article data from a DOM element"""
        try:
            title = ''
            article_url = ''

            # Extract title
            title_selectors = ['h1', 'h2', 'h3', 'h4', '.headline', '.title', '.story-title']
            for selector in title_selectors:
                title_elem = element.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break

            # Extract URL
            link_elem = element.select_one('a[href]')
            if link_elem:
                article_url = link_elem.get('href')
            elif element.name == 'a' and element.get('href'):
                article_url = element.get('href')
                if not title:
                    title = element.get_text().strip()

            if not title or not article_url or len(title) < 10:
                return None

            # Make URL absolute
            if article_url.startswith('/'):
                article_url = base_url + article_url
            elif not article_url.startswith('http'):
                return None

            # Scrape article content
            content = self._scrape_article_content(article_url)
            if not content or len(content) < 100:
                return None

            return {
                'title': title,
                'content': content,
                'url': article_url
            }

        except Exception as e:
            print(f"Error extracting article: {e}")
            return None

    def _calculate_popularity_score(self, element: BeautifulSoup) -> int:
        """Calculate popularity score based on element indicators"""
        score = 1  # Base score

        # Check for popularity indicators in classes/attributes
        element_html = str(element)
        popularity_indicators = [
            ('trending', 8), ('popular', 8), ('most-read', 9), ('top-story', 9),
            ('featured', 7), ('breaking', 10), ('headline', 6), ('main', 7),
            ('hero', 8), ('lead', 7)
        ]

        for indicator, bonus in popularity_indicators:
            if indicator in element_html.lower():
                score += bonus
                break

        # Check position (earlier = more popular)
        parent = element.parent
        if parent:
            siblings = parent.find_all(['article', 'div'], class_=True)
            position = 0
            for i, sibling in enumerate(siblings):
                if sibling == element:
                    position = i
                    break

            # Higher score for earlier position
            if position == 0:
                score += 5
            elif position <= 2:
                score += 3
            elif position <= 5:
                score += 1

        return score

# Streamlit cache for analyzer instance
@st.cache_resource
def get_news_analyzer():
    """Get cached news analyzer instance"""
    return LocalNewsAnalyzer()