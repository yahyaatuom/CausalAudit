# llm_interface.py
import os
import time
import json
import re
import sys
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional

# Dynamic path detection
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class GroqLLM:
    def __init__(self, model="llama-3.1-8b-versatile"):
        """
        Initialize Groq LLM client with structured output support.
        
        Raises:
            ValueError: If GROQ_API_KEY is missing
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = model
        self.temperature = 0
        self.max_retries = 2
        
        if not self.api_key:
            error_msg = (
                "\n" + "="*60 + "\n"
                "❌ FATAL ERROR: GROQ_API_KEY not found in environment\n"
                "="*60 + "\n"
                "Please set your Groq API key in the .env file:\n"
                "   GROQ_API_KEY=gsk_your_api_key_here\n\n"
                "Get your API key from: https://console.groq.com/keys\n"
                "="*60 + "\n"
            )
            print(error_msg)
            raise ValueError("GROQ_API_KEY is required but not found. Check your .env file.")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def _build_structured_prompt(self, scenario_description: str) -> str:
        """Build prompt that forces structured JSON output."""
        return f"""You are a causal reasoning system for urban transportation incidents.

INCIDENT: {scenario_description}

TASK: Analyze the incident and output a STRICT JSON object with EXACTLY these fields.
Do NOT include any text before or after the JSON. Do NOT use markdown code blocks.

REQUIRED JSON STRUCTURE:
{{
    "primary_cause": "string (one sentence describing the main cause)",
    "mechanism": "string (causal chain using → arrows, e.g., 'A → B → C')",
    "contributing_factors": ["factor1", "factor2", "factor3"],
    "temporal_sequence": ["event1 at time1", "event2 at time2", "event3 at time3"],
    "spatial_location": "string (specific road or intersection)",
    "confidence": 0.95
}}

CAUSAL REASONING RULES:
1. Cause must precede effect in time (temporal precedence)
2. Cause must be geographically plausible (spatial relevance)
3. Causal mechanism must be physically possible (mechanistic plausibility)
4. Avoid spurious correlations (correlation ≠ causation)
5. Include all necessary causal factors (completeness)

EXAMPLE RESPONSE:
{{
    "primary_cause": "Hydroplaning due to standing water on wet road",
    "mechanism": "heavy rain → standing water accumulation → tire hydroplaning → loss of steering control → collision with barrier",
    "contributing_factors": ["high speed", "insufficient following distance", "reduced visibility"],
    "temporal_sequence": ["6:42 PM: heavy rain begins", "6:42 PM: standing water forms", "6:42 PM: van hydroplanes", "6:42 PM: chain reaction collision"],
    "spatial_location": "Sheikh Zayed Road southbound near Al Khail interchange",
    "confidence": 0.95
}}

Now respond with ONLY valid JSON (no other text):"""
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start > end:
            return None
        
        json_str = text[start:end+1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    
    def generate_explanation(self, scenario_description: str) -> Dict[str, Any]:
        """Send scenario to Groq and get structured explanation with retry logic."""
        prompt = self._build_structured_prompt(scenario_description)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=800,
                    timeout=45
                )
                
                raw_response = response.choices[0].message.content
                structured = self._extract_json_from_response(raw_response)
                
                if structured and all(k in structured for k in ["primary_cause", "mechanism"]):
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
                    # Handle case where JSON extraction failed but we got a response
                    return {
                        'structured_output': {
                            "primary_cause": "PARSE_ERROR: Could not extract JSON",
                            "mechanism": "unknown",
                            "contributing_factors": [],
                            "temporal_sequence": [],
                            "spatial_location": "unknown",
                            "confidence": 0.0
                        },
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
                    return {
                        'structured_output': {
                            "primary_cause": f"API_ERROR: {str(e)[:100]}",
                            "mechanism": "unknown",
                            "contributing_factors": [],
                            "temporal_sequence": [],
                            "spatial_location": "unknown",
                            "confidence": 0.0
                        },
                        'explanation': f"Error: {str(e)}",
                        'model': self.model,
                        'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
                        'structured': False,
                        'error': str(e)
                    }
                time.sleep(2)
        
        # Fallback if all retries exhausted
        return {
            'structured_output': {
                "primary_cause": "MAX_RETRIES_EXCEEDED",
                "mechanism": "unknown",
                "contributing_factors": [],
                "temporal_sequence": [],
                "spatial_location": "unknown",
                "confidence": 0.0
            },
            'explanation': "Error after max retries",
            'model': self.model,
            'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
            'structured': False,
            'error': "Max retries exceeded"
        }
    
    def set_model(self, model_name: str):
        """Change the model after initialization."""
        self.model = model_name


# For backward compatibility
NVIDIALLM = GroqLLM