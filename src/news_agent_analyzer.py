"""
News Agent Analyzer
Uses Azure AI Agent to analyze scraped news articles for topics and sentiment
"""

import json
from typing import Dict, List, Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder


class NewsAgentAnalyzer:
    """Analyzes news articles using Azure AI Agent"""

    def __init__(self):
        """Initialize Azure AI client and agent"""
        try:
            self.project = AIProjectClient(
                credential=DefaultAzureCredential(),
                endpoint="https://shared-ai-hub-foundry.services.ai.azure.com/api/projects/Team-15-Communities"
            )
            self.agent_id = "asst_hpw4j8I1zrSDVIFidlfegvSi"
            self.agent = self.project.agents.get_agent(self.agent_id)
        except Exception as e:
            print(f"Warning: Could not initialize Azure AI Agent: {e}")
            self.project = None
            self.agent = None

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """
        Analyze a list of articles using the AI agent

        Args:
            articles: List of article dictionaries with title, content, url

        Returns:
            Analysis results including topics and sentiment
        """
        if not self.project or not self.agent:
            return self._fallback_analysis(articles)

        if not articles:
            return {'error': 'No articles provided for analysis'}

        try:
            # Prepare articles for analysis
            articles_text = self._prepare_articles_for_analysis(articles)

            # Create thread
            thread = self.project.agents.threads.create()

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(articles_text)

            # Send message to agent
            message = self.project.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=analysis_prompt
            )

            # Run analysis
            run = self.project.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=self.agent_id
            )

            if run.status == "failed":
                print(f"Agent run failed: {run.last_error}")
                return self._fallback_analysis(articles)

            # Get response
            messages = self.project.agents.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING
            )

            # Extract agent response
            agent_responses = []
            for message in messages:
                if message.role == "assistant" and message.text_messages:
                    agent_responses.append(message.text_messages[-1].text.value)

            if agent_responses:
                return self._parse_agent_response(agent_responses[-1], articles)
            else:
                return self._fallback_analysis(articles)

        except Exception as e:
            print(f"Error during agent analysis: {e}")
            return self._fallback_analysis(articles)

    def _prepare_articles_for_analysis(self, articles: List[Dict]) -> str:
        """Prepare articles text for AI analysis"""
        articles_summary = []

        for i, article in enumerate(articles[:10], 1):  # Limit to top 10 articles
            title = article.get('title', 'No title')
            content = article.get('content', '')
            source = article.get('source', 'Unknown source')

            # Truncate content to avoid token limits
            content_preview = content[:500] + '...' if len(content) > 500 else content

            article_summary = f"""
Article {i}:
Title: {title}
Source: {source}
Content: {content_preview}
---
"""
            articles_summary.append(article_summary)

        return '\n'.join(articles_summary)

    def _create_analysis_prompt(self, articles_text: str) -> str:
        """Create analysis prompt for the AI agent"""
        return f"""
Please analyze the following local news articles and provide a comprehensive analysis in JSON format.

I need you to analyze these articles for:
1. Main community topics (housing, transport, education, healthcare, crime/safety, economy, environment, local government)
2. Overall sentiment and trust indicators
3. Key themes and concerns
4. Community cohesion insights

Articles to analyze:
{articles_text}

Please respond with a JSON object containing:
{{
    "main_topics": [list of main topics found],
    "topic_analysis": {{
        "housing": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "transport": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "education": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "healthcare": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "crime_safety": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "economy": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "environment": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}},
        "local_government": {{"mentions": 0, "sentiment": "neutral", "key_issues": []}}
    }},
    "overall_sentiment": {{"score": 0.0, "description": "neutral", "confidence": "medium"}},
    "trust_indicators": {{"positive": [], "negative": [], "overall_trust_level": "medium"}},
    "community_cohesion": {{"score": 5.0, "indicators": [], "recommendations": []}},
    "key_themes": [],
    "summary": "Brief summary of the analysis"
}}

Focus on local community issues and how they might affect social cohesion and trust.
"""

    def _parse_agent_response(self, response: str, articles: List[Dict]) -> Dict:
        """Parse agent response into structured format"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed_response = json.loads(json_str)

                # Add metadata
                parsed_response['agent_analysis'] = True
                parsed_response['articles_analyzed'] = len(articles)
                parsed_response['analysis_method'] = 'azure_ai_agent'

                return parsed_response
            else:
                # If no valid JSON found, create structured response from text
                return {
                    'agent_analysis': True,
                    'articles_analyzed': len(articles),
                    'analysis_method': 'azure_ai_agent_fallback',
                    'raw_response': response,
                    'summary': response[:500] + '...' if len(response) > 500 else response,
                    'main_topics': self._extract_topics_from_text(response),
                    'overall_sentiment': {'score': 0.0, 'description': 'neutral', 'confidence': 'low'}
                }

        except json.JSONDecodeError:
            return self._fallback_analysis(articles)

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from unstructured text"""
        topics = []
        topic_keywords = {
            'housing': ['housing', 'homes', 'property', 'rent'],
            'transport': ['transport', 'traffic', 'bus', 'train'],
            'education': ['school', 'education', 'university'],
            'healthcare': ['health', 'hospital', 'NHS', 'medical'],
            'crime_safety': ['crime', 'police', 'safety', 'security'],
            'economy': ['business', 'jobs', 'economy', 'employment'],
            'environment': ['environment', 'green', 'pollution'],
            'local_government': ['council', 'mayor', 'government', 'policy']
        }

        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _fallback_analysis(self, articles: List[Dict]) -> Dict:
        """Fallback analysis if agent is unavailable"""
        if not articles:
            return {'error': 'No articles to analyze'}

        # Simple keyword-based analysis
        all_text = ' '.join([
            f"{article.get('title', '')} {article.get('content', '')}"
            for article in articles
        ]).lower()

        # Topic analysis
        topic_keywords = {
            'housing': ['housing', 'homes', 'property', 'rent', 'development'],
            'transport': ['transport', 'traffic', 'bus', 'train', 'road'],
            'education': ['school', 'education', 'university', 'student'],
            'healthcare': ['health', 'hospital', 'NHS', 'medical', 'doctor'],
            'crime_safety': ['crime', 'police', 'safety', 'security', 'theft'],
            'economy': ['business', 'jobs', 'economy', 'employment', 'shop'],
            'environment': ['environment', 'green', 'pollution', 'climate'],
            'local_government': ['council', 'mayor', 'government', 'policy']
        }

        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            topic_scores[topic] = score

        main_topics = [topic for topic, score in topic_scores.items() if score > 0]
        main_topics.sort(key=lambda x: topic_scores[x], reverse=True)

        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve']
        negative_words = ['bad', 'terrible', 'negative', 'problem', 'issue', 'concern']

        positive_count = sum(1 for word in positive_words if word in all_text)
        negative_count = sum(1 for word in negative_words if word in all_text)

        if positive_count > negative_count:
            sentiment = {'score': 0.3, 'description': 'slightly positive', 'confidence': 'low'}
        elif negative_count > positive_count:
            sentiment = {'score': -0.3, 'description': 'slightly negative', 'confidence': 'low'}
        else:
            sentiment = {'score': 0.0, 'description': 'neutral', 'confidence': 'low'}

        return {
            'agent_analysis': False,
            'articles_analyzed': len(articles),
            'analysis_method': 'fallback_keyword',
            'main_topics': main_topics[:5],
            'topic_scores': topic_scores,
            'overall_sentiment': sentiment,
            'summary': f'Analyzed {len(articles)} articles covering topics: {", ".join(main_topics[:3])}'
        }

    def analyze_single_article(self, article: Dict) -> Dict:
        """Analyze a single article"""
        return self.analyze_articles([article])