"""
AWS Bedrock Claude client for document processing.
Replaces OpenAI integration with Claude 3.5 Haiku.
"""

import json
import boto3
import base64
import re
from typing import Dict, Any, List, Optional

from ..config.settings import BEDROCK_REGION, BEDROCK_MODEL_ID, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


def strip_json_code_fences(s: str) -> str:
    """Remove JSON code fences from string."""
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl+1:]
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


class BedrockClaudeClient:
    """
    AWS Bedrock client for Claude 3.5 Haiku.
    Provides drop-in replacement for OpenAI functionality.
    """
    
    def __init__(self, region: str = None, model_id: str = None):
        self.region = region or BEDROCK_REGION
        self.model_id = model_id or BEDROCK_MODEL_ID
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            # Use explicit credentials if provided, otherwise use default AWS credential chain
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                )
            else:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region
                )
        return self._client
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        temperature: float = 0,
        max_tokens: int = 4096
    ) -> str:
        """
        Send a chat completion request to Claude via Bedrock.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text from Claude
        """
        # Build request body for Claude
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system:
            body["system"] = system
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    
    def chat_with_image(
        self,
        system: str,
        user_text: str,
        image_data: str,
        media_type: str = "image/jpeg",
        temperature: float = 0,
        max_tokens: int = 4096
    ) -> str:
        """
        Send a vision request with an image to Claude.
        
        Args:
            system: System prompt
            user_text: User message text
            image_data: Base64 image data or data URI
            media_type: MIME type of image (image/jpeg, image/png, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text from Claude
        """
        # Handle base64 data URI format
        if image_data.startswith("data:"):
            # Extract base64 from data URI: data:image/jpeg;base64,/9j/4AAQ...
            match = re.match(r'data:([^;]+);base64,(.+)', image_data)
            if match:
                media_type = match.group(1)
                image_data = match.group(2)
            else:
                # Fallback: just split on comma
                parts = image_data.split(",", 1)
                if len(parts) == 2:
                    image_data = parts[1]
                    if "image/png" in parts[0]:
                        media_type = "image/png"
                    elif "image/gif" in parts[0]:
                        media_type = "image/gif"
                    elif "image/webp" in parts[0]:
                        media_type = "image/webp"
        
        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ]
        
        return self.chat_completion(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def chat_json(
        self,
        system: str,
        user_payload: dict,
        temperature: float = 0,
        max_tokens: int = 4096
    ) -> dict:
        """
        Get a JSON response from Claude.
        Equivalent to OpenAI's response_format={"type": "json_object"}
        
        Args:
            system: System prompt
            user_payload: Dict to send as JSON in user message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON dict (empty dict on error)
        """
        system_msg = (
            system
            + "\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown code fences, no explanations, no extra text. Just the raw JSON."
        )
        
        user_msg = (
            "You MUST return a single JSON object only. No prose, no code fences, no markdown.\n\n"
            "Payload follows as JSON:\n"
            + json.dumps(user_payload, ensure_ascii=False)
        )
        
        messages = [{"role": "user", "content": user_msg}]
        
        try:
            response = self.chat_completion(
                messages=messages,
                system=system_msg,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Clean and parse JSON
            response = strip_json_code_fences(response)
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[BedrockClaudeClient.chat_json] JSON parse error: {e}")
            print(f"[BedrockClaudeClient.chat_json] Raw response: {response[:500]}")
            return {}
        except Exception as e:
            print(f"[BedrockClaudeClient.chat_json ERROR] {e}")
            return {}
    
    def chat_json_with_image(
        self,
        system: str,
        user_text: str,
        image_data: str,
        media_type: str = "image/jpeg",
        temperature: float = 0,
        max_tokens: int = 4096
    ) -> dict:
        """
        Get a JSON response from Claude with an image input.
        
        Args:
            system: System prompt
            user_text: User message text
            image_data: Base64 image data or data URI
            media_type: MIME type of image
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON dict (empty dict on error)
        """
        system_msg = (
            system
            + "\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown code fences, no explanations, no extra text. Just the raw JSON."
        )
        
        try:
            response = self.chat_with_image(
                system=system_msg,
                user_text=user_text + "\n\nReturn ONLY valid JSON.",
                image_data=image_data,
                media_type=media_type,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Clean and parse JSON
            response = strip_json_code_fences(response)
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[BedrockClaudeClient.chat_json_with_image] JSON parse error: {e}")
            print(f"[BedrockClaudeClient.chat_json_with_image] Raw response: {response[:500] if response else 'None'}")
            return {}
        except Exception as e:
            print(f"[BedrockClaudeClient.chat_json_with_image ERROR] {e}")
            return {}


# Global client instance (lazy initialization)
_client: Optional[BedrockClaudeClient] = None


def get_bedrock_client() -> BedrockClaudeClient:
    """Get or create the global Bedrock client."""
    global _client
    if _client is None:
        _client = BedrockClaudeClient()
    return _client


# Convenience functions for drop-in replacement

def chat_completion(
    messages: List[Dict[str, Any]],
    system: str = "",
    temperature: float = 0,
    max_tokens: int = 4096
) -> str:
    """Convenience function for chat completion."""
    return get_bedrock_client().chat_completion(
        messages=messages,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens
    )


def chat_with_image(
    system: str,
    user_text: str,
    image_data: str,
    media_type: str = "image/jpeg",
    temperature: float = 0,
    max_tokens: int = 4096
) -> str:
    """Convenience function for vision requests."""
    return get_bedrock_client().chat_with_image(
        system=system,
        user_text=user_text,
        image_data=image_data,
        media_type=media_type,
        temperature=temperature,
        max_tokens=max_tokens
    )


def chat_json(
    system: str,
    user_payload: dict,
    temperature: float = 0,
    max_tokens: int = 4096
) -> dict:
    """Convenience function for JSON responses."""
    return get_bedrock_client().chat_json(
        system=system,
        user_payload=user_payload,
        temperature=temperature,
        max_tokens=max_tokens
    )


def chat_json_with_image(
    system: str,
    user_text: str,
    image_data: str,
    media_type: str = "image/jpeg",
    temperature: float = 0,
    max_tokens: int = 4096
) -> dict:
    """Convenience function for JSON responses with images."""
    return get_bedrock_client().chat_json_with_image(
        system=system,
        user_text=user_text,
        image_data=image_data,
        media_type=media_type,
        temperature=temperature,
        max_tokens=max_tokens
    )


