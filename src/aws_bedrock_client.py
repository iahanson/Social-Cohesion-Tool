"""
AWS Bedrock Client
Handles interactions with AWS Bedrock for GenAI text analysis using Claude Sonnet 4
"""

import json
import os
import boto3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock"""
    region: str
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int

class AWSBedrockClient:
    """Client for AWS Bedrock Claude Sonnet 4"""
    
    def __init__(self):
        self.config = self._load_config()
        self.bedrock_client = self._create_client()
        
    def _load_config(self) -> BedrockConfig:
        """Load AWS Bedrock configuration from environment variables"""
        # Check for required AWS credentials
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables")
        
        if access_key.startswith('BedrockAPIKey') or secret_key.startswith('ABSKQ'):
            print("⚠️ Warning: AWS credentials appear to be placeholder values")
            print("Please update your .env file with real AWS credentials")
        
        return BedrockConfig(
            region=os.getenv('AWS_BEDROCK_REGION', 'us-east-1'),
            model_id=os.getenv('AWS_BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0'),
            max_tokens=int(os.getenv('AWS_BEDROCK_MAX_TOKENS', '4000')),
            temperature=float(os.getenv('AWS_BEDROCK_TEMPERATURE', '0.1')),
            top_p=float(os.getenv('AWS_BEDROCK_TOP_P', '0.9')),
            top_k=int(os.getenv('AWS_BEDROCK_TOP_K', '250'))
        )
    
    def _create_client(self):
        """Create AWS Bedrock client"""
        try:
            # Get credentials
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            session_token = os.getenv('AWS_SESSION_TOKEN')
            
            # Initialize Bedrock client for runtime operations
            client_kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': self.config.region,
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            }
            
            if session_token and session_token != 'your_session_token':
                client_kwargs['aws_session_token'] = session_token
            
            client = boto3.client(**client_kwargs)
            
            # Test the connection using a separate bedrock client
            self._test_connection()
            print(f"✅ AWS Bedrock client initialized (Region: {self.config.region})")
            return client
            
        except Exception as e:
            print(f"❌ Error initializing AWS Bedrock client: {e}")
            raise
    
    def _test_connection(self):
        """Test connection to AWS Bedrock"""
        try:
            # Get credentials
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            session_token = os.getenv('AWS_SESSION_TOKEN')
            
            # Create a separate bedrock client for listing models
            client_kwargs = {
                'service_name': 'bedrock',
                'region_name': self.config.region,
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            }
            
            if session_token and session_token != 'your_session_token':
                client_kwargs['aws_session_token'] = session_token
            
            bedrock_client = boto3.client(**client_kwargs)
            
            # Simple test to verify access
            response = bedrock_client.list_foundation_models()
            print(f"✅ AWS Bedrock connection successful")
        except Exception as e:
            print(f"⚠️ AWS Bedrock connection test failed: {e}")
            # Don't raise here, let the actual API call handle it
    
    def generate_text(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate text using Claude Sonnet 4
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with response data
        """
        try:
            # Prepare the request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "messages": []
            }
            
            # Add system message if provided
            if system_prompt:
                body["messages"].append({
                    "role": "user",
                    "content": f"System: {system_prompt}\n\nUser: {prompt}"
                })
            else:
                body["messages"].append({
                    "role": "user",
                    "content": prompt
                })
            
            # Make the API call
            response = self.bedrock_client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            return {
                'success': True,
                'content': response_body['content'][0]['text'],
                'usage': response_body.get('usage', {}),
                'model': self.config.model_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': self.config.model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Claude Sonnet 4 (via text generation)
        
        Note: Claude doesn't have a direct embedding model, so we'll use
        a workaround by generating semantic representations
        """
        try:
            embeddings = []
            
            for text in texts:
                # Use Claude to generate a semantic representation
                prompt = f"""
                Analyze the following text and provide a structured semantic representation:
                
                Text: "{text}"
                
                Please provide:
                1. Key themes and topics
                2. Sentiment indicators
                3. Social cohesion relevance
                4. Geographic references
                5. Severity indicators
                
                Format as a structured analysis.
                """
                
                response = self.generate_text(prompt)
                
                if response['success']:
                    # Convert the structured response to a numerical representation
                    # This is a simplified approach - in practice you might want to use
                    # a dedicated embedding model
                    embedding = self._text_to_embedding(response['content'])
                    embeddings.append(embedding)
                else:
                    # Fallback to zero vector
                    embeddings.append([0.0] * 384)  # Standard embedding dimension
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 384] * len(texts)
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """
        Convert text to numerical embedding
        This is a simplified approach - in production you'd want to use
        a proper embedding model like Amazon Titan Embeddings
        """
        # Simple hash-based embedding for demonstration
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert to numerical values
        embedding = []
        for i in range(0, len(text_hash), 2):
            val = int(text_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
            embedding.append(val)
        
        # Pad or truncate to standard dimension
        target_dim = 384
        if len(embedding) < target_dim:
            embedding.extend([0.0] * (target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]
        
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            embeddings = self.generate_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                return 0.0
            
            # Calculate cosine similarity
            import numpy as np
            
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_id': self.config.model_id,
            'region': self.config.region,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k
        }
    
    def test_connection(self) -> bool:
        """Test the connection to AWS Bedrock"""
        try:
            test_prompt = "Hello, this is a test message."
            response = self.generate_text(test_prompt)
            return response['success']
        except Exception:
            return False
