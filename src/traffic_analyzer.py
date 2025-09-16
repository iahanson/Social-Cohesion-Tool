"""
Traffic Analyzer for Article URLs
Determines the most viewed article from a list of URLs using various traffic estimation methods
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
from collections import Counter
import re
import json


class ArticleTrafficAnalyzer:
    """Analyzes traffic for article URLs to determine most viewed content"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Traffic estimation weights
        self.traffic_indicators = {
            'social_shares': 5,
            'comments': 4,
            'page_views': 10,
            'time_recency': 3,
            'homepage_prominence': 8,
            'social_media_mentions': 6,
            'read_time': 2
        }

    def analyze_article_traffic(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze traffic for a list of articles and rank by popularity

        Args:
            articles: List of article dicts with 'url', 'title', 'content' keys

        Returns:
            Articles sorted by estimated traffic/popularity with traffic_score added
        """
        if not articles:
            return articles

        scored_articles = []

        for article in articles:
            try:
                traffic_score = self._calculate_traffic_score(article)
                article_with_score = article.copy()
                article_with_score['traffic_score'] = traffic_score
                article_with_score['traffic_indicators'] = self._get_traffic_indicators(article)
                scored_articles.append(article_with_score)
            except Exception as e:
                print(f"Error analyzing traffic for {article.get('url', 'unknown')}: {e}")
                article_with_score = article.copy()
                article_with_score['traffic_score'] = 0
                article_with_score['traffic_indicators'] = {}
                scored_articles.append(article_with_score)

        # Sort by traffic score (highest first)
        scored_articles.sort(key=lambda x: x['traffic_score'], reverse=True)
        return scored_articles

    def get_most_popular_article(self, articles: List[Dict]) -> Optional[Dict]:
        """Get the most popular article from the list"""
        if not articles:
            return None

        analyzed_articles = self.analyze_article_traffic(articles)
        return analyzed_articles[0] if analyzed_articles else None

    def _calculate_traffic_score(self, article: Dict) -> float:
        """Calculate overall traffic score for an article"""
        url = article.get('url', '')
        title = article.get('title', '')
        content = article.get('content', '')

        if not url:
            return 0

        score = 0

        # Base score from existing popularity_score if available
        base_score = article.get('popularity_score', 1)
        score += base_score

        # Analyze URL structure for popularity indicators
        score += self._analyze_url_structure(url)

        # Analyze content length and quality indicators
        score += self._analyze_content_quality(content)

        # Analyze title for engagement indicators
        score += self._analyze_title_engagement(title)

        # Try to get social media and engagement signals
        score += self._get_social_engagement_score(url)

        # Time-based scoring (newer articles often have more traffic initially)
        score += self._get_recency_bonus(article)

        return round(score, 2)

    def _analyze_url_structure(self, url: str) -> float:
        """Analyze URL structure for traffic indicators"""
        score = 0
        url_lower = url.lower()

        # URLs with certain patterns often indicate popular content
        popular_patterns = [
            'breaking', 'urgent', 'exclusive', 'live', 'update',
            'major', 'important', 'crisis', 'emergency'
        ]

        for pattern in popular_patterns:
            if pattern in url_lower:
                score += 2

        # Shorter URLs often indicate more important content
        path_length = len(urlparse(url).path)
        if path_length < 50:
            score += 1
        elif path_length > 100:
            score -= 1

        return score

    def _analyze_content_quality(self, content: str) -> float:
        """Analyze content for quality and engagement indicators"""
        if not content:
            return 0

        score = 0
        content_lower = content.lower()
        word_count = len(content.split())

        # Optimal length articles often get more engagement
        if 300 <= word_count <= 1200:
            score += 2
        elif word_count > 1200:
            score += 1

        # Look for engagement keywords
        engagement_keywords = [
            'breaking', 'exclusive', 'revealed', 'shocking', 'amazing',
            'incredible', 'urgent', 'important', 'major', 'significant'
        ]

        for keyword in engagement_keywords:
            if keyword in content_lower:
                score += 1

        # Look for social/community indicators
        community_indicators = [
            'residents', 'local', 'community', 'neighbours', 'families',
            'people', 'public', 'citizens'
        ]

        community_mentions = sum(1 for indicator in community_indicators if indicator in content_lower)
        score += min(community_mentions * 0.5, 3)  # Cap at 3 points

        return score

    def _analyze_title_engagement(self, title: str) -> float:
        """Analyze title for engagement potential"""
        if not title:
            return 0

        score = 0
        title_lower = title.lower()

        # Headlines with numbers often perform better
        if re.search(r'\d+', title):
            score += 1

        # Question headlines often get engagement
        if '?' in title:
            score += 1

        # Emotional/action words
        engagement_words = [
            'new', 'first', 'last', 'best', 'worst', 'top', 'biggest',
            'major', 'huge', 'massive', 'shock', 'surprise', 'revealed'
        ]

        for word in engagement_words:
            if word in title_lower:
                score += 0.5

        # Optimal title length
        if 10 <= len(title.split()) <= 15:
            score += 1

        return score

    def _get_social_engagement_score(self, url: str) -> float:
        """Try to estimate social engagement (simplified version)"""
        # This is a simplified version - in production you might use
        # social media APIs or third-party services
        score = 0

        try:
            # Get the page content to look for social share counts
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for social share indicators in the page
                share_indicators = soup.find_all(text=re.compile(r'\d+\s*(shares?|comments?|likes?)', re.I))
                if share_indicators:
                    # Extract numbers and sum them
                    numbers = []
                    for indicator in share_indicators:
                        nums = re.findall(r'\d+', indicator)
                        numbers.extend([int(n) for n in nums])

                    if numbers:
                        total_engagement = sum(numbers)
                        # Convert to logarithmic score to avoid huge numbers
                        score = min(5, total_engagement ** 0.3)

                time.sleep(0.1)  # Be respectful with requests

        except Exception:
            pass  # Fail silently for social engagement

        return score

    def _get_recency_bonus(self, article: Dict) -> float:
        """Give bonus points to more recent articles"""
        # Since we don't have publish dates, use position in scraping
        # as proxy for recency (first scraped = more prominent = likely newer)

        # This is a simplification - in a real system you'd parse publish dates
        popularity_score = article.get('popularity_score', 1)

        # Articles with higher popularity_score were likely found first/more prominent
        if popularity_score >= 8:
            return 2
        elif popularity_score >= 5:
            return 1
        else:
            return 0

    def _get_traffic_indicators(self, article: Dict) -> Dict:
        """Get detailed traffic indicators for an article"""
        indicators = {}

        url = article.get('url', '')
        title = article.get('title', '')
        content = article.get('content', '')

        indicators['content_quality'] = self._analyze_content_quality(content)
        indicators['title_engagement'] = self._analyze_title_engagement(title)
        indicators['url_indicators'] = self._analyze_url_structure(url)
        indicators['base_popularity'] = article.get('popularity_score', 1)
        indicators['recency_bonus'] = self._get_recency_bonus(article)

        return indicators

    def extract_article_image(self, url: str) -> Optional[str]:
        """Extract the main image from an article URL"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find Open Graph image first
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image['content']
                # Make absolute URL if needed
                if image_url.startswith('/'):
                    parsed_url = urlparse(url)
                    image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{image_url}"
                return image_url

            # Try Twitter card image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                image_url = twitter_image['content']
                if image_url.startswith('/'):
                    parsed_url = urlparse(url)
                    image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{image_url}"
                return image_url

            # Look for article images
            article_selectors = [
                'article img', '.article-content img', '.entry-content img',
                '.story-body img', '.post-content img', '.content img'
            ]

            for selector in article_selectors:
                img = soup.select_one(selector)
                if img and img.get('src'):
                    image_url = img['src']
                    if image_url.startswith('/'):
                        parsed_url = urlparse(url)
                        image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{image_url}"
                    return image_url

            # Fallback: look for any reasonably sized image
            images = soup.find_all('img')
            for img in images:
                src = img.get('src', '')
                if src and not any(skip in src.lower() for skip in ['logo', 'icon', 'avatar', 'thumb']):
                    if src.startswith('/'):
                        parsed_url = urlparse(url)
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}{src}"
                    return src

        except Exception as e:
            print(f"Error extracting image from {url}: {e}")

        return None

    def get_article_metadata(self, url: str) -> Dict:
        """Extract metadata from article URL including image and social data"""
        metadata = {
            'title': '',
            'description': '',
            'image': '',
            'site_name': '',
            'url': url
        }

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return metadata

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract Open Graph data
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            for tag in og_tags:
                prop = tag.get('property', '').replace('og:', '')
                content = tag.get('content', '')
                if prop and content:
                    metadata[prop] = content

            # Extract Twitter Card data as fallback
            twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            for tag in twitter_tags:
                name = tag.get('name', '').replace('twitter:', '')
                content = tag.get('content', '')
                if name and content and not metadata.get(name):
                    metadata[name] = content

            # Extract basic meta tags
            if not metadata.get('title'):
                title_tag = soup.find('title')
                if title_tag:
                    metadata['title'] = title_tag.get_text().strip()

            if not metadata.get('description'):
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                if desc_tag:
                    metadata['description'] = desc_tag.get('content', '')

            # Extract image if not found
            if not metadata.get('image'):
                metadata['image'] = self.extract_article_image(url)

        except Exception as e:
            print(f"Error extracting metadata from {url}: {e}")

        return metadata