#!/usr/bin/env python3
"""
News Scraper and Analyzer
Scrapes local news websites and analyzes for community topics and social trust sentiment
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse
import json
from collections import Counter
from news_sources_finder import NewsSourcesFinder

# Simple sentiment analysis without external dependencies
class SimpleSentimentAnalyzer:
    def __init__(self):
        # Community and trust-related keywords
        self.positive_trust_words = [
            'community', 'together', 'cooperation', 'unity', 'support', 'collaboration',
            'partnership', 'teamwork', 'solidarity', 'mutual', 'trust', 'help',
            'volunteer', 'charity', 'kindness', 'generous', 'caring', 'friendly',
            'welcome', 'inclusive', 'celebrate', 'festival', 'success', 'achievement'
        ]

        self.negative_trust_words = [
            'conflict', 'division', 'dispute', 'tension', 'mistrust', 'disagreement',
            'controversy', 'scandal', 'corruption', 'fraud', 'crime', 'violence',
            'protest', 'anger', 'frustration', 'concern', 'worry', 'fear',
            'problem', 'issue', 'crisis', 'struggle', 'difficulty', 'challenge'
        ]

        self.community_topics = {
            'housing': ['housing', 'homes', 'rent', 'landlord', 'tenant', 'property', 'development'],
            'transport': ['transport', 'bus', 'train', 'traffic', 'parking', 'road', 'cycle'],
            'education': ['school', 'education', 'teacher', 'student', 'university', 'college'],
            'healthcare': ['hospital', 'health', 'NHS', 'medical', 'doctor', 'care', 'treatment'],
            'crime_safety': ['police', 'crime', 'safety', 'security', 'theft', 'antisocial'],
            'economy': ['jobs', 'employment', 'business', 'economy', 'shop', 'market', 'money'],
            'environment': ['environment', 'green', 'pollution', 'climate', 'park', 'nature'],
            'local_government': ['council', 'mayor', 'election', 'policy', 'planning', 'budget']
        }

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment focusing on community trust indicators"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_count = sum(1 for word in words if word in self.positive_trust_words)
        negative_count = sum(1 for word in words if word in self.negative_trust_words)

        # Calculate scores
        total_words = len(words)
        if total_words == 0:
            return {'sentiment_score': 0, 'trust_score': 0, 'word_count': 0}

        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words

        # Sentiment score: -1 (very negative) to 1 (very positive)
        sentiment_score = (positive_ratio - negative_ratio) * 2
        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]

        # Trust score: 0 (low trust) to 10 (high trust)
        if positive_count + negative_count > 0:
            trust_score = (positive_count / (positive_count + negative_count)) * 10
        else:
            trust_score = 5.0  # Neutral if no trust indicators

        return {
            'sentiment_score': round(sentiment_score, 3),
            'trust_score': round(trust_score, 1),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'word_count': total_words
        }

    def extract_topics(self, text: str) -> Dict[str, int]:
        """Extract community-related topics from text"""
        text_lower = text.lower()
        topic_counts = {}

        for topic, keywords in self.community_topics.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            topic_counts[topic] = count

        return topic_counts

class NewsArticle:
    def __init__(self, title: str, content: str, url: str, publish_date: Optional[datetime] = None):
        self.title = title
        self.content = content
        self.url = url
        self.publish_date = publish_date
        self.sentiment_analysis = None
        self.topics = None

class NewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.analyzer = SimpleSentimentAnalyzer()

    def scrape_website(self, url: str, max_articles: int = 5) -> List[NewsArticle]:
        """Scrape articles from a news website"""
        articles = []

        try:
            print(f"🔍 Scraping: {url}")

            # Get main page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article links
            article_links = self._find_article_links(soup, url)

            print(f"📄 Found {len(article_links)} potential articles")

            # Scrape individual articles
            scraped_count = 0
            for article_url, link_text in article_links[:max_articles * 3]:  # Try more than needed
                if scraped_count >= max_articles:
                    break

                article = self._scrape_article(article_url, link_text)
                if article:
                    articles.append(article)
                    scraped_count += 1

                # Be respectful - add delay
                time.sleep(1)

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

        return articles

    def _find_article_links(self, soup: BeautifulSoup, base_url: str) -> List[Tuple[str, str]]:
        """Find article links on a webpage"""
        links = []

        # Common selectors for article links
        selectors = [
            'a[href*="article"]',
            'a[href*="news"]',
            'a[href*="story"]',
            'a[href*="/20"]',  # Date-based URLs
            'article a',
            '.article a',
            '.news-item a',
            '.story a',
            'h1 a',
            'h2 a',
            'h3 a'
        ]

        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    link_text = elem.get_text().strip()

                    # Filter out unwanted links
                    if self._is_valid_article_link(full_url, link_text):
                        links.append((full_url, link_text))

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link[0] not in seen:
                seen.add(link[0])
                unique_links.append(link)

        return unique_links[:20]  # Limit to 20 potential articles

    def _is_valid_article_link(self, url: str, text: str) -> bool:
        """Check if a link is likely to be a news article"""
        url_lower = url.lower()
        text_lower = text.lower()

        # Skip certain types of links
        skip_patterns = [
            'javascript:', 'mailto:', '#', 'contact', 'about', 'privacy',
            'terms', 'subscribe', 'newsletter', 'advertisement', 'cookie'
        ]

        for pattern in skip_patterns:
            if pattern in url_lower or pattern in text_lower:
                return False

        # Must have reasonable text length
        if len(text) < 10 or len(text) > 200:
            return False

        return True

    def _scrape_article(self, url: str, fallback_title: str = "") -> Optional[NewsArticle]:
        """Scrape a single article"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = self._extract_title(soup) or fallback_title

            # Extract content
            content = self._extract_content(soup)

            # Skip if content is too short
            if not content or len(content) < 200:
                return None

            # Try to extract publish date
            publish_date = self._extract_publish_date(soup)

            return NewsArticle(title, content, url, publish_date)

        except Exception as e:
            print(f"  ⚠️  Failed to scrape {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        selectors = ['h1', 'title', '.article-title', '.entry-title', '.headline']

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if len(title) > 10:
                    return title
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()

        # Try different content selectors
        selectors = [
            '.article-content', '.entry-content', '.post-content',
            'article', '.content', '.story-body', '.article-body',
            '[class*="content"]', '[class*="article"]'
        ]

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                content = elem.get_text()
                content = ' '.join(content.split())  # Clean whitespace
                if len(content) > 200:
                    return content

        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        content = ' '.join(content.split())  # Clean whitespace

        return content

    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Try to extract publish date"""
        date_selectors = ['time', '[datetime]', '.date', '.published', '.post-date']

        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_str = elem.get('datetime') or elem.get_text()
                try:
                    # Simple date parsing - could be enhanced
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    pass
        return None

    def analyze_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze articles for sentiment and topics"""
        for article in articles:
            combined_text = f"{article.title} {article.content}"
            article.sentiment_analysis = self.analyzer.analyze_sentiment(combined_text)
            article.topics = self.analyzer.extract_topics(combined_text)

        return articles

class NewsAnalyzer:
    def __init__(self):
        self.finder = NewsSourcesFinder()
        self.scraper = NewsScraper()

    def analyze_area_news(self, lat: float, lon: float, radius_km: float = 30, max_articles_per_source: int = 3) -> Dict:
        """Complete analysis of news in a geographic area"""
        print(f"🎯 Analyzing news around coordinates ({lat}, {lon}) within {radius_km}km")

        # Find nearby news sources
        sources = self.finder.search_nearby(lat, lon, radius_km)

        if not sources:
            return {'error': f'No news sources found within {radius_km}km'}

        print(f"📰 Found {len(sources)} news sources")

        all_articles = []
        source_results = {}

        # Scrape each source
        for source in sources[:5]:  # Limit to 5 sources for demo
            source_name = source['name']
            source_url = source['url']

            print(f"\n📋 Processing: {source_name}")

            articles = self.scraper.scrape_website(source_url, max_articles_per_source)
            analyzed_articles = self.scraper.analyze_articles(articles)

            all_articles.extend(analyzed_articles)
            source_results[source_name] = {
                'articles_found': len(analyzed_articles),
                'url': source_url,
                'distance_km': source.get('distance_km', 0)
            }

            print(f"  ✅ Scraped {len(analyzed_articles)} articles")

        # Generate analysis report
        return self._generate_analysis_report(all_articles, source_results, lat, lon, radius_km)

    def _generate_analysis_report(self, articles: List[NewsArticle], source_results: Dict,
                                lat: float, lon: float, radius_km: float) -> Dict:
        """Generate comprehensive analysis report"""
        if not articles:
            return {'error': 'No articles were successfully scraped'}

        # Overall sentiment analysis
        sentiment_scores = [a.sentiment_analysis['sentiment_score'] for a in articles]
        trust_scores = [a.sentiment_analysis['trust_score'] for a in articles]

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_trust = sum(trust_scores) / len(trust_scores)

        # Topic analysis
        all_topics = {}
        for article in articles:
            for topic, count in article.topics.items():
                all_topics[topic] = all_topics.get(topic, 0) + count

        # Sort topics by frequency
        sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)

        # Sentiment distribution
        positive_articles = len([s for s in sentiment_scores if s > 0.1])
        negative_articles = len([s for s in sentiment_scores if s < -0.1])
        neutral_articles = len(sentiment_scores) - positive_articles - negative_articles

        # Generate recommendations
        recommendations = self._generate_recommendations(avg_sentiment, avg_trust, sorted_topics)

        return {
            'analysis_summary': {
                'coordinates': (lat, lon),
                'radius_km': radius_km,
                'articles_analyzed': len(articles),
                'sources_scraped': len(source_results),
                'analysis_date': datetime.now().isoformat()
            },
            'sentiment_analysis': {
                'average_sentiment': round(avg_sentiment, 3),
                'average_trust_score': round(avg_trust, 1),
                'positive_articles': positive_articles,
                'negative_articles': negative_articles,
                'neutral_articles': neutral_articles
            },
            'topic_analysis': {
                'top_topics': sorted_topics[:10],
                'all_topics': all_topics
            },
            'source_results': source_results,
            'articles': [
                {
                    'title': a.title[:100] + '...' if len(a.title) > 100 else a.title,
                    'url': a.url,
                    'sentiment_score': a.sentiment_analysis['sentiment_score'],
                    'trust_score': a.sentiment_analysis['trust_score'],
                    'word_count': a.sentiment_analysis['word_count'],
                    'top_topics': sorted(a.topics.items(), key=lambda x: x[1], reverse=True)[:3]
                }
                for a in articles[:10]  # Show details for first 10 articles
            ],
            'recommendations': recommendations
        }

    def _generate_recommendations(self, avg_sentiment: float, avg_trust: float,
                                top_topics: List[Tuple[str, int]]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if avg_sentiment < -0.2:
            recommendations.append("Community sentiment appears negative - consider initiatives to address local concerns")
        elif avg_sentiment > 0.2:
            recommendations.append("Positive community sentiment detected - good foundation for community building")

        if avg_trust < 4.0:
            recommendations.append("Low trust indicators - focus on transparency and community engagement")
        elif avg_trust > 7.0:
            recommendations.append("High trust indicators - leverage for collaborative community projects")

        # Topic-based recommendations
        if top_topics:
            top_topic = top_topics[0][0]
            if top_topic == 'crime_safety':
                recommendations.append("Safety concerns prominent - consider community safety initiatives")
            elif top_topic == 'housing':
                recommendations.append("Housing issues frequently discussed - housing support may be needed")
            elif top_topic == 'transport':
                recommendations.append("Transport issues highlighted - infrastructure improvements may help")
            elif top_topic == 'healthcare':
                recommendations.append("Healthcare topics prevalent - health services engagement recommended")

        if not recommendations:
            recommendations.append("Continue monitoring local news for emerging community issues")

        return recommendations

def main():
    """Main function to demonstrate news analysis"""
    analyzer = NewsAnalyzer()

    # Example coordinates - Manchester
    lat = 53.4808
    lon = -2.2426

    print("🔍 Starting News Analysis for Local Community Trust and Sentiment")
    print("=" * 70)

    try:
        results = analyzer.analyze_area_news(lat, lon, radius_km=50, max_articles_per_source=2)

        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return

        # Print results
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 50)

        summary = results['analysis_summary']
        print(f"📍 Location: ({summary['coordinates'][0]}, {summary['coordinates'][1]})")
        print(f"📏 Radius: {summary['radius_km']}km")
        print(f"📰 Articles: {summary['articles_analyzed']}")
        print(f"🏢 Sources: {summary['sources_scraped']}")

        sentiment = results['sentiment_analysis']
        print(f"\n💭 SENTIMENT ANALYSIS:")
        print(f"  Average Sentiment: {sentiment['average_sentiment']} (-1 to 1 scale)")
        print(f"  Trust Score: {sentiment['average_trust_score']}/10")
        print(f"  Article Distribution:")
        print(f"    Positive: {sentiment['positive_articles']}")
        print(f"    Neutral: {sentiment['neutral_articles']}")
        print(f"    Negative: {sentiment['negative_articles']}")

        topics = results['topic_analysis']
        print(f"\n🏷️ TOP COMMUNITY TOPICS:")
        for topic, count in topics['top_topics'][:5]:
            print(f"  {topic.replace('_', ' ').title()}: {count} mentions")

        print(f"\n📄 SAMPLE ARTICLES:")
        for i, article in enumerate(results['articles'][:3], 1):
            print(f"  {i}. {article['title']}")
            print(f"     Sentiment: {article['sentiment_score']}, Trust: {article['trust_score']}")
            print(f"     Topics: {', '.join([t[0] for t in article['top_topics'][:2]])}")

        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")

        # Save results
        with open(f'news_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to JSON file")

    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()