# llm_interface.py
import os
import time
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional

# Dynamic path detection
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class GroqLLM:
    def __init__(self, model="llama-3.3-70b-versatile"):
        """
        Initialize Groq LLM client with structured output support
        
        Available models on Groq (free tier):
        - llama-3.3-70b-versatile (best quality, good speed)
        - llama-3.1-8b-instant (fastest)
        - mixtral-8x7b-32768 (good for complex reasoning)
        - gemma2-9b-it (lightweight)
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = model
        self.temperature = 0
        self.max_retries = 2
        
        print(f"DEBUG: API Key loaded: {'Yes' if self.api_key else 'No'}")
        print(f"DEBUG: Using model: {self.model}")
        
        if not self.api_key:
            print("⚠️ Error: GROQ_API_KEY not found. Check your .env file.")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
    
    def _build_structured_prompt(self, scenario_description: str) -> str:
        """
        Build prompt that forces structured JSON output
        """
        return f"""Analyze the following incident and provide a structured causal explanation.

INCIDENT: {scenario_description}

You MUST respond with a valid JSON object containing exactly these fields:

{{
    "primary_cause": "one sentence describing the main cause",
    "mechanism": "step-by-step causal chain using → arrows (e.g., 'event A → event B → event C')",
    "contributing_factors": ["factor1", "factor2", "factor3"],
    "temporal_sequence": ["event1 at time1", "event2 at time2"],
    "spatial_location": "specific road or intersection where incident occurred",
    "confidence": 0.0-1.0 (how confident are you in this explanation)
}}

RULES:
- Do NOT include any text outside the JSON object
- Do NOT wrap in markdown code blocks (just raw JSON)
- Use double quotes, not single quotes
- Ensure valid JSON syntax

Example response for a weather-related crash:
{{
    "primary_cause": "Hydroplaning due to standing water on wet road",
    "mechanism": "heavy rain → standing water accumulation → tire hydroplaning → loss of steering control → collision with barrier",
    "contributing_factors": ["high speed", "insufficient following distance", "reduced visibility"],
    "temporal_sequence": ["6:42 PM: heavy rain begins", "6:42 PM: standing water forms", "6:42 PM: van hydroplanes", "6:42 PM: chain reaction collision"],
    "spatial_location": "Sheikh Zayed Road southbound near Al Khail interchange",
    "confidence": 0.95
}}

Now respond with ONLY valid JSON:"""
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response, handling various edge cases
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None
        
        json_str = json_match.group()
        
        # Try to fix common JSON issues
        # Replace single quotes with double quotes (careful)
        try:
            # First attempt: direct parse
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Second attempt: fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Third attempt: use ast.literal_eval as fallback
                import ast
                try:
                    # Convert JSON-like string to Python dict
                    result = ast.literal_eval(json_str)
                    if isinstance(result, dict):
                        return result
                except (SyntaxError, ValueError):
                    pass
        
        return None
    
    def _get_fallback_response(self, scenario_description: str, error_msg: str = "") -> Dict[str, Any]:
        """
        Return structured fallback when API or parsing fails
        """
        return {
            'structured_output': {
                "primary_cause": f"Unable to analyze: {error_msg[:100]}",
                "mechanism": "analysis_failed",
                "contributing_factors": [],
                "temporal_sequence": [],
                "spatial_location": "unknown",
                "confidence": 0.0
            },
            'explanation': f"Analysis temporarily unavailable. {error_msg}\n\nIncident: {scenario_description[:200]}...",
            'model': self.model,
            'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
            'structured': False,
            'error': error_msg
        }
    
    def generate_explanation(self, scenario_description: str) -> Dict[str, Any]:
        """
        Send scenario to Groq and get structured explanation with retry logic
        """
        if not self.client:
            return self._get_fallback_response(scenario_description, "API key not configured")
        
        prompt = self._build_structured_prompt(scenario_description)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500,
                    timeout=45
                )
                
                raw_response = response.choices[0].message.content
                
                # Extract structured JSON
                structured = self._extract_json_from_response(raw_response)
                
                if structured and all(k in structured for k in ["primary_cause", "mechanism"]):
                    # Valid structured output
                    return {
                        'structured_output': structured,
                        'explanation': raw_response,
                        'model': self.model,
                        'tokens': {
                            'prompt': response.usage.prompt_tokens,
                            'completion': response.usage.completion_tokens,
                            'total': response.usage.total_tokens
                        },
                        'structured': True,
                        'error': None
                    }
                else:
                    # Parsing failed but response received
                    return {
                        'structured_output': None,
                        'explanation': raw_response,
                        'model': self.model,
                        'tokens': {
                            'prompt': response.usage.prompt_tokens,
                            'completion': response.usage.completion_tokens,
                            'total': response.usage.total_tokens
                        },
                        'structured': False,
                        'error': "Failed to parse JSON response"
                    }
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    return self._get_fallback_response(scenario_description, str(e))
                time.sleep(2)
        
        return self._get_fallback_response(scenario_description, "Max retries exceeded")
    
    def set_model(self, model_name: str):
        """Change the model after initialization"""
        self.model = model_name
        print(f"✅ Model changed to: {self.model}")
    
    def get_structured_explanation(self, scenario_description: str) -> Optional[Dict[str, Any]]:
        """
        Convenience method to get only the structured output
        """
        result = self.generate_explanation(scenario_description)
        return result.get('structured_output')


# For backward compatibility with existing code
NVIDIALLM = GroqLLM