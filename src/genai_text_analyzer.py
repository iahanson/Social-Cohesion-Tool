"""
GenAI Text Analyzer for Social Cohesion
Supports both Azure OpenAI GPT-4.1-mini and AWS Bedrock Claude Sonnet 4
to analyze text for social cohesion issues and identify localities for MSOA mapping
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
from openai import AzureOpenAI
import pandas as pd
from dotenv import load_dotenv
from .aws_bedrock_client import AWSBedrockClient

# Load environment variables from .env file
load_dotenv(override=True)

@dataclass
class SocialCohesionIssue:
    """Represents a social cohesion issue found in text"""
    issue_type: str
    severity: str  # Low, Medium, High, Critical
    description: str
    confidence: float  # 0-1
    location_mentioned: Optional[str] = None
    msoa_code: Optional[str] = None
    local_authority: Optional[str] = None
    keywords: List[str] = None
    context: str = ""

@dataclass
class TextAnalysisResult:
    """Result of text analysis"""
    text_id: str
    source: str
    timestamp: datetime
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SocialCohesionIssue]
    localities_found: List[Dict[str, str]]
    summary: str
    recommendations: List[str]

class GenAITextAnalyzer:
    """GenAI-powered text analyzer for social cohesion issues"""
    
    def __init__(self):
        # Determine which provider to use
        self.provider = os.getenv("GENAI_PROVIDER", "azure").lower()
        print(f"🔍 Detected provider: {self.provider}")
        
        if self.provider == "aws":
            try:
                # Initialize AWS Bedrock client
                self.bedrock_client = AWSBedrockClient()
                self.client = self.bedrock_client
                self.model = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
                print("✅ Using AWS Bedrock Claude Sonnet 4")
            except Exception as e:
                print(f"❌ Failed to initialize AWS Bedrock: {e}")
                print("🔄 Falling back to Azure OpenAI")
                self.provider = "azure"
                self.client = self._initialize_azure_openai()
                self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
                self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
                print("✅ Using Azure OpenAI GPT-4.1-mini (fallback)")
        else:
            # Initialize Azure OpenAI client (default)
            self.client = self._initialize_azure_openai()
            self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
            self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
            print("✅ Using Azure OpenAI GPT-4.1-mini")
        
        # Load locality mapping data
        self.locality_mapping = self._load_locality_mapping()
        
        # Define social cohesion issue categories
        self.issue_categories = {
            "community_tension": {
                "keywords": ["conflict", "tension", "dispute", "argument", "hostility", "animosity"],
                "description": "Community tensions and conflicts"
            },
            "social_isolation": {
                "keywords": ["isolated", "lonely", "excluded", "marginalized", "left out", "alone"],
                "description": "Social isolation and exclusion"
            },
            "discrimination": {
                "keywords": ["discrimination", "racism", "prejudice", "bias", "unfair", "targeted"],
                "description": "Discrimination and prejudice"
            },
            "economic_stress": {
                "keywords": ["poverty", "unemployment", "struggling", "financial", "money problems", "cost of living"],
                "description": "Economic stress and financial hardship"
            },
            "housing_issues": {
                "keywords": ["housing", "homeless", "eviction", "overcrowded", "poor conditions", "rent"],
                "description": "Housing-related issues"
            },
            "crime_safety": {
                "keywords": ["crime", "unsafe", "violence", "theft", "vandalism", "antisocial"],
                "description": "Crime and safety concerns"
            },
            "service_access": {
                "keywords": ["services", "access", "transport", "healthcare", "education", "cutbacks"],
                "description": "Access to services and facilities"
            },
            "environmental": {
                "keywords": ["pollution", "environment", "green space", "air quality", "noise", "litter"],
                "description": "Environmental and quality of life issues"
            }
        }
    
    def _initialize_azure_openai(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            return client
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    def _load_locality_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load locality to MSOA mapping data"""
        # This would typically load from a comprehensive locality database
        # For now, we'll use a simplified mapping
        return {
            "london": {
                "boroughs": {
                    "kensington and chelsea": "E02000001",
                    "westminster": "E02000002",
                    "hammersmith and fulham": "E02000003",
                    "wandsworth": "E02000004",
                    "lambeth": "E02000005",
                    "southwark": "E02000006",
                    "tower hamlets": "E02000007",
                    "hackney": "E02000008",
                    "islington": "E02000009",
                    "camden": "E02000010"
                }
            },
            "postcodes": {
                "sw1": "E02000001",  # Kensington and Chelsea
                "sw3": "E02000001",
                "sw5": "E02000001",
                "sw7": "E02000001",
                "sw10": "E02000001",
                "w1": "E02000002",   # Westminster
                "w2": "E02000002",
                "w8": "E02000002",
                "w9": "E02000002",
                "w10": "E02000002"
            }
        }
    
    def analyze_text(self, text: str, source: str = "unknown", text_id: str = None) -> TextAnalysisResult:
        """
        Analyze text for social cohesion issues using GenAI
        
        Args:
            text: Text to analyze
            source: Source of the text (e.g., "survey", "social_media", "report")
            text_id: Unique identifier for the text
            
        Returns:
            TextAnalysisResult object with analysis results
        """
        if not text_id:
            text_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Step 1: Use GenAI to analyze the text
            ai_analysis = self._analyze_with_genai(text)
            
            # Step 2: Extract localities and map to MSOAs
            localities = self._extract_localities(text, ai_analysis)
            
            # Step 3: Process issues and assign localities
            issues = self._process_issues(ai_analysis, localities)
            
            # Step 4: Generate summary and recommendations
            summary = self._generate_summary(issues)
            recommendations = self._generate_recommendations(issues)
            
            # Step 5: Count issues by severity
            severity_counts = self._count_issues_by_severity(issues)
            
            return TextAnalysisResult(
                text_id=text_id,
                source=source,
                timestamp=datetime.now(),
                total_issues=len(issues),
                critical_issues=severity_counts["Critical"],
                high_issues=severity_counts["High"],
                medium_issues=severity_counts["Medium"],
                low_issues=severity_counts["Low"],
                issues=issues,
                localities_found=localities,
                summary=summary,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            raise
    
    def _analyze_with_genai(self, text: str) -> Dict[str, Any]:
        """Use GenAI (Azure OpenAI or AWS Bedrock) to analyze text for social cohesion issues"""
        
        system_prompt = """You are an expert social cohesion analyst. Your task is to analyze text for social cohesion issues and identify localities mentioned.

Analyze the following text and identify:
1. Social cohesion issues (community tension, social isolation, discrimination, economic stress, housing issues, crime/safety, service access, environmental)
2. Severity level (Low, Medium, High, Critical)
3. Confidence level (0-1)
4. Specific localities mentioned (areas, boroughs, postcodes, landmarks)
5. Keywords that indicate the issue
6. Context around each issue

Return your analysis as a JSON object with this structure:
{
    "issues": [
        {
            "issue_type": "community_tension",
            "severity": "High",
            "confidence": 0.8,
            "description": "Brief description of the issue",
            "keywords": ["keyword1", "keyword2"],
            "context": "Relevant context from the text",
            "location_mentioned": "specific area mentioned"
        }
    ],
    "localities": [
        {
            "name": "area name",
            "type": "borough/postcode/landmark",
            "context": "how it was mentioned"
        }
    ],
    "overall_assessment": "Brief overall assessment of social cohesion"
}"""

        user_prompt = f"Please analyze this text for social cohesion issues:\n\n{text}"
        
        try:
            if self.provider == "aws":
                # Use AWS Bedrock
                response = self.bedrock_client.generate_text(user_prompt, system_prompt)
                
                if not response['success']:
                    raise Exception(f"AWS Bedrock error: {response['error']}")
                
                ai_response = response['content']
                
            else:
                # Use Azure OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                ai_response = response.choices[0].message.content
            
            # Parse the response
            content = ai_response
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                return {
                    "issues": [],
                    "localities": [],
                    "overall_assessment": "Analysis completed but JSON parsing failed"
                }
                
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            return {
                "issues": [],
                "localities": [],
                "overall_assessment": f"Error in analysis: {str(e)}"
            }
    
    def _extract_localities(self, text: str, ai_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract localities from text and AI analysis"""
        localities = []
        
        # Get localities from AI analysis
        if "localities" in ai_analysis:
            for locality in ai_analysis["localities"]:
                localities.append({
                    "name": locality.get("name", ""),
                    "type": locality.get("type", ""),
                    "context": locality.get("context", ""),
                    "msoa_code": self._map_locality_to_msoa(locality.get("name", ""))
                })
        
        # Also extract postcodes from text
        postcode_pattern = r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b'
        postcodes = re.findall(postcode_pattern, text.upper())
        
        for postcode in postcodes:
            clean_postcode = postcode.replace(" ", "").upper()
            msoa_code = self._map_postcode_to_msoa(clean_postcode)
            if msoa_code:
                localities.append({
                    "name": postcode,
                    "type": "postcode",
                    "context": "mentioned in text",
                    "msoa_code": msoa_code
                })
        
        return localities
    
    def _map_locality_to_msoa(self, locality_name: str) -> Optional[str]:
        """Map locality name to MSOA code"""
        if not locality_name:
            return None
        
        locality_lower = locality_name.lower().strip()
        
        # Check boroughs
        for borough, msoa_code in self.locality_mapping["london"]["boroughs"].items():
            if borough in locality_lower or locality_lower in borough:
                return msoa_code
        
        # Check postcodes
        for postcode_prefix, msoa_code in self.locality_mapping["postcodes"].items():
            if locality_lower.startswith(postcode_prefix.lower()):
                return msoa_code
        
        return None
    
    def _map_postcode_to_msoa(self, postcode: str) -> Optional[str]:
        """Map postcode to MSOA code"""
        if not postcode:
            return None
        
        # Extract postcode prefix (e.g., SW1 from SW1A 1AA)
        postcode_prefix = re.match(r'^[A-Z]{1,2}\d{1,2}', postcode)
        if postcode_prefix:
            prefix = postcode_prefix.group().lower()
            return self.locality_mapping["postcodes"].get(prefix)
        
        return None
    
    def _process_issues(self, ai_analysis: Dict[str, Any], localities: List[Dict[str, str]]) -> List[SocialCohesionIssue]:
        """Process issues from AI analysis"""
        issues = []
        
        if "issues" not in ai_analysis:
            return issues
        
        for issue_data in ai_analysis["issues"]:
            # Find associated locality
            location_mentioned = issue_data.get("location_mentioned", "")
            msoa_code = None
            local_authority = None
            
            if location_mentioned:
                msoa_code = self._map_locality_to_msoa(location_mentioned)
                if msoa_code:
                    local_authority = self._get_local_authority_from_msoa(msoa_code)
            
            issue = SocialCohesionIssue(
                issue_type=issue_data.get("issue_type", "unknown"),
                severity=issue_data.get("severity", "Low"),
                description=issue_data.get("description", ""),
                confidence=float(issue_data.get("confidence", 0.5)),
                location_mentioned=location_mentioned,
                msoa_code=msoa_code,
                local_authority=local_authority,
                keywords=issue_data.get("keywords", []),
                context=issue_data.get("context", "")
            )
            issues.append(issue)
        
        return issues
    
    def _get_local_authority_from_msoa(self, msoa_code: str) -> Optional[str]:
        """Get local authority name from MSOA code"""
        # This would typically query a database or use a mapping service
        # For now, we'll use a simplified mapping
        msoa_to_la = {
            "E02000001": "Kensington and Chelsea",
            "E02000002": "Westminster",
            "E02000003": "Hammersmith and Fulham",
            "E02000004": "Wandsworth",
            "E02000005": "Lambeth",
            "E02000006": "Southwark",
            "E02000007": "Tower Hamlets",
            "E02000008": "Hackney",
            "E02000009": "Islington",
            "E02000010": "Camden"
        }
        return msoa_to_la.get(msoa_code)
    
    def _count_issues_by_severity(self, issues: List[SocialCohesionIssue]) -> Dict[str, int]:
        """Count issues by severity level"""
        counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for issue in issues:
            counts[issue.severity] += 1
        return counts
    
    def _generate_summary(self, issues: List[SocialCohesionIssue]) -> str:
        """Generate a summary of the analysis"""
        if not issues:
            return "No social cohesion issues identified in the text."
        
        total_issues = len(issues)
        critical_issues = len([i for i in issues if i.severity == "Critical"])
        high_issues = len([i for i in issues if i.severity == "High"])
        
        summary = f"Analysis identified {total_issues} social cohesion issues"
        if critical_issues > 0:
            summary += f", including {critical_issues} critical issues"
        if high_issues > 0:
            summary += f" and {high_issues} high-priority issues"
        summary += "."
        
        return summary
    
    def _generate_recommendations(self, issues: List[SocialCohesionIssue]) -> List[str]:
        """Generate recommendations based on identified issues"""
        recommendations = []
        
        if not issues:
            return ["Continue monitoring for social cohesion indicators."]
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate recommendations for each issue type
        for issue_type, type_issues in issue_types.items():
            if issue_type == "community_tension":
                recommendations.append("Consider community mediation services and conflict resolution programs.")
            elif issue_type == "social_isolation":
                recommendations.append("Implement community engagement programs and social connection initiatives.")
            elif issue_type == "discrimination":
                recommendations.append("Review and strengthen anti-discrimination policies and awareness programs.")
            elif issue_type == "economic_stress":
                recommendations.append("Provide economic support services and financial advice programs.")
            elif issue_type == "housing_issues":
                recommendations.append("Address housing quality and availability through targeted interventions.")
            elif issue_type == "crime_safety":
                recommendations.append("Enhance community safety measures and crime prevention programs.")
            elif issue_type == "service_access":
                recommendations.append("Improve access to essential services and community facilities.")
            elif issue_type == "environmental":
                recommendations.append("Address environmental concerns and improve local environmental quality.")
        
        # Add general recommendations
        if len(issues) > 5:
            recommendations.append("Consider comprehensive community development approach due to multiple issues.")
        
        return recommendations
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Azure OpenAI
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.provider == "aws":
                # Use AWS Bedrock for embeddings
                return self.bedrock_client.generate_embeddings(texts)
            else:
                # Use Azure OpenAI for embeddings
                embeddings = []
                for text in texts:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
                return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if self.provider == "aws":
                # Use AWS Bedrock for embeddings
                embeddings = self.bedrock_client.generate_embeddings([text])
                return embeddings[0] if embeddings else []
            else:
                # Use Azure OpenAI for embeddings
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if self.provider == "aws":
                # Use AWS Bedrock similarity calculation
                return self.bedrock_client.calculate_similarity(text1, text2)
            else:
                # Use Azure OpenAI embeddings for similarity
                # Generate embeddings for both texts
                embedding1 = self.generate_single_embedding(text1)
                embedding2 = self.generate_single_embedding(text2)
                
                # Calculate cosine similarity
                import numpy as np
                
                # Convert to numpy arrays
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_issues(self, target_text: str, existing_issues: List[SocialCohesionIssue], threshold: float = 0.7) -> List[Tuple[SocialCohesionIssue, float]]:
        """
        Find similar issues to a target text
        
        Args:
            target_text: Text to find similar issues for
            existing_issues: List of existing issues to compare against
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of tuples (issue, similarity_score) above threshold
        """
        try:
            similar_issues = []
            
            for issue in existing_issues:
                # Combine issue description and context for comparison
                issue_text = f"{issue.description} {issue.context}"
                similarity = self.calculate_text_similarity(target_text, issue_text)
                
                if similarity >= threshold:
                    similar_issues.append((issue, similarity))
            
            # Sort by similarity score (highest first)
            similar_issues.sort(key=lambda x: x[1], reverse=True)
            return similar_issues
            
        except Exception as e:
            print(f"Error finding similar issues: {e}")
            return []
    
    def analyze_multiple_texts(self, texts: List[Tuple[str, str, str]]) -> List[TextAnalysisResult]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of tuples (text, source, text_id)
            
        Returns:
            List of TextAnalysisResult objects
        """
        results = []
        for text, source, text_id in texts:
            try:
                result = self.analyze_text(text, source, text_id)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing text {text_id}: {e}")
                continue
        
        return results
    
    def export_results(self, results: List[TextAnalysisResult], format: str = "json") -> str:
        """
        Export analysis results
        
        Args:
            results: List of TextAnalysisResult objects
            format: Export format ("json", "csv", "summary")
            
        Returns:
            Exported data as string
        """
        if format == "json":
            return json.dumps([self._result_to_dict(result) for result in results], indent=2, default=str)
        elif format == "csv":
            return self._results_to_csv(results)
        elif format == "summary":
            return self._results_to_summary(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _result_to_dict(self, result: TextAnalysisResult) -> Dict[str, Any]:
        """Convert TextAnalysisResult to dictionary"""
        return {
            "text_id": result.text_id,
            "source": result.source,
            "timestamp": result.timestamp.isoformat(),
            "total_issues": result.total_issues,
            "critical_issues": result.critical_issues,
            "high_issues": result.high_issues,
            "medium_issues": result.medium_issues,
            "low_issues": result.low_issues,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "description": issue.description,
                    "location_mentioned": issue.location_mentioned,
                    "msoa_code": issue.msoa_code,
                    "local_authority": issue.local_authority,
                    "keywords": issue.keywords,
                    "context": issue.context
                }
                for issue in result.issues
            ],
            "localities_found": result.localities_found,
            "summary": result.summary,
            "recommendations": result.recommendations
        }
    
    def _results_to_csv(self, results: List[TextAnalysisResult]) -> str:
        """Convert results to CSV format"""
        rows = []
        for result in results:
            for issue in result.issues:
                rows.append({
                    "text_id": result.text_id,
                    "source": result.source,
                    "timestamp": result.timestamp.isoformat(),
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "description": issue.description,
                    "location_mentioned": issue.location_mentioned,
                    "msoa_code": issue.msoa_code,
                    "local_authority": issue.local_authority,
                    "keywords": ", ".join(issue.keywords) if issue.keywords else "",
                    "context": issue.context
                })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    def _results_to_summary(self, results: List[TextAnalysisResult]) -> str:
        """Convert results to summary format"""
        summary = []
        summary.append("=== GenAI Text Analysis Summary ===\n")
        
        total_texts = len(results)
        total_issues = sum(result.total_issues for result in results)
        total_critical = sum(result.critical_issues for result in results)
        total_high = sum(result.high_issues for result in results)
        
        summary.append(f"Total texts analyzed: {total_texts}")
        summary.append(f"Total issues identified: {total_issues}")
        summary.append(f"Critical issues: {total_critical}")
        summary.append(f"High-priority issues: {total_high}")
        summary.append("")
        
        for result in results:
            summary.append(f"--- {result.text_id} ({result.source}) ---")
            summary.append(f"Timestamp: {result.timestamp}")
            summary.append(f"Issues: {result.total_issues} (Critical: {result.critical_issues}, High: {result.high_issues})")
            summary.append(f"Summary: {result.summary}")
            summary.append("")
        
        return "\n".join(summary)
