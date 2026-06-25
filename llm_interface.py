# llm_interface.py
"""
LLM Interface with multiple API keys, model switching, and rate limit handling.
"""

import os
import time
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# Dynamic path detection
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class GroqLLM:
    """Groq LLM client with multi-key rotation and model switching."""
    
    
    MODELS = {
    'high_quality': 'llama-3.3-70b-versatile',  # Still active
    'balanced': 'llama-3.1-8b-instant',         # Active, fast
    'fast': 'gemma2-9b-it',                      # Active, efficient
    # 'mixtral-8x7b-32768' has been REMOVED - decommissioned
}
    
    # Token estimates per scenario
    TOKEN_ESTIMATES = {
        'llama-3.3-70b-versatile': 1200,
        'mixtral-8x7b-32768': 1000,
        'llama-3.1-8b-instant': 600,
        'gemma2-9b-it': 800
    }
    
    def __init__(self, model="llama-3.1-8b-instant", max_retries=3):
        """
        Initialize Groq LLM client with multiple API keys.
        
        Args:
            model: Default model to use
            max_retries: Maximum retry attempts per request
            
        Raises:
            ValueError: If no valid API keys found
        """
        self.model = model
        self.max_retries = max_retries
        self.temperature = 0
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            error_msg = (
                "\n" + "="*60 + "\n"
                "❌ FATAL ERROR: No valid GROQ API keys found\n"
                "="*60 + "\n"
                "Please set at least one API key in the .env file:\n"
                "   GROQ_API_KEY_1=gsk_your_api_key_here\n"
                "   GROQ_API_KEY_2=gsk_your_second_key_here\n"
                "   GROQ_API_KEY_3=gsk_your_third_key_here\n\n"
                "Or simply: GROQ_API_KEY=gsk_your_api_key_here\n"
                "Get your API keys from: https://console.groq.com/keys\n"
                "="*60 + "\n"
            )
            print(error_msg)
            raise ValueError("No valid GROQ API keys found")
        
        self.current_key_index = 0
        self._init_client()
        
        # Token tracking
        self.token_usage = 0
        self.daily_limit = 100000
        self.reset_time = None
        
        # Model switching
        self.current_model = model
        self.model_fallback_used = False
        
        print(f"✅ Initialized GroqLLM with {len(self.api_keys)} API keys")
        print(f"   Default model: {self.current_model}")
    
    def _load_api_keys(self) -> List[str]:
        """Load all available API keys from environment."""
        keys = []
        
        # Try multiple key formats
        for i in range(1, 10):  # Support up to 9 keys
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and key.strip():
                keys.append(key.strip())
        
        # Fallback to single key
        if not keys:
            single_key = os.getenv("GROQ_API_KEY")
            if single_key and single_key.strip():
                keys.append(single_key.strip())
        
        return keys
    
    def _init_client(self):
        """Initialize OpenAI client with current API key."""
        key = self.api_keys[self.current_key_index]
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=key
        )
    
    def _rotate_key(self):
        """Rotate to next available API key."""
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._init_client()
        print(f"🔄 Rotated API key: {old_index+1} → {self.current_key_index+1}")
        time.sleep(1)  # Brief pause after rotation
    
    def _switch_model(self, direction='down'):
        """Switch to a different model (up or down in quality)."""
        model_chain = ['high_quality', 'balanced', 'fast', 'gemma']
        current_level = None
        
        for level, name in self.MODELS.items():
            if name == self.current_model:
                current_level = level
                break
        
        if not current_level:
            return
        
        if direction == 'down':
            # Move to faster/lower quality model
            idx = model_chain.index(current_level)
            if idx < len(model_chain) - 1:
                new_level = model_chain[idx + 1]
                self.current_model = self.MODELS[new_level]
                print(f"📉 Switched to smaller model: {self.current_model}")
                self.model_fallback_used = True
        else:
            # Move to higher quality model
            idx = model_chain.index(current_level)
            if idx > 0:
                new_level = model_chain[idx - 1]
                self.current_model = self.MODELS[new_level]
                print(f"📈 Switched to larger model: {self.current_model}")
    
    def _get_token_estimate(self, model: str = None) -> int:
        """Get estimated tokens per scenario for a model."""
        model = model or self.current_model
        return self.TOKEN_ESTIMATES.get(model, 1000)
    
    def can_process(self, num_scenarios: int, model: str = None) -> bool:
        """Check if enough tokens remain to process N scenarios."""
        model = model or self.current_model
        estimate = self._get_token_estimate(model)
        needed = num_scenarios * estimate
        remaining = self.daily_limit - self.token_usage
        return remaining > needed
    
    def _record_usage(self, tokens_used: int):
        """Record token usage and check limits."""
        self.token_usage += tokens_used
        remaining = self.daily_limit - self.token_usage
        
        # Print warnings
        if remaining < 10000:
            print(f"⚠️ Approaching rate limit! {remaining} tokens remaining")
        if remaining < 5000:
            print(f"🚨 CRITICAL: Only {remaining} tokens left!")
        
        # Save usage to file for persistence
        self._save_usage()
    
    def _save_usage(self):
        """Save token usage to file for persistence across runs."""
        usage_file = Path(__file__).parent / "token_usage.json"
        data = {
            'token_usage': self.token_usage,
            'daily_limit': self.daily_limit,
            'last_updated': time.time(),
            'model': self.current_model
        }
        with open(usage_file, 'w') as f:
            json.dump(data, f)
    
    def _load_usage(self):
        """Load token usage from file."""
        usage_file = Path(__file__).parent / "token_usage.json"
        if usage_file.exists():
            try:
                with open(usage_file, 'r') as f:
                    data = json.load(f)
                    # Reset if more than 24 hours have passed
                    if time.time() - data.get('last_updated', 0) > 86400:
                        self.token_usage = 0
                        print("🔄 Daily token usage reset")
                    else:
                        self.token_usage = data.get('token_usage', 0)
                        print(f"📊 Loaded token usage: {self.token_usage}/{self.daily_limit}")
            except:
                pass
    
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
        # Load usage from previous runs
        self._load_usage()
        
        prompt = self._build_structured_prompt(scenario_description)
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=800,
                    timeout=45
                )
                
                raw_response = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
                # Record usage
                self._record_usage(tokens_used)
                
                structured = self._extract_json_from_response(raw_response)
                
                if structured and all(k in structured for k in ["primary_cause", "mechanism"]):
                    return {
                        'structured_output': structured,
                        'explanation': raw_response,
                        'model': self.current_model,
                        'tokens': {
                            'prompt': response.usage.prompt_tokens,
                            'completion': response.usage.completion_tokens,
                            'total': tokens_used
                        },
                        'structured': True,
                        'error': None
                    }
                else:
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
                        'model': self.current_model,
                        'tokens': {
                            'prompt': response.usage.prompt_tokens,
                            'completion': response.usage.completion_tokens,
                            'total': tokens_used
                        },
                        'structured': False,
                        'error': "Failed to parse JSON response"
                    }
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                # Handle rate limit errors
                if "rate_limit" in error_msg.lower():
                    # Try switching to a smaller model
                    self._switch_model('down')
                    
                    # Try rotating API key
                    if len(self.api_keys) > 1:
                        self._rotate_key()
                    
                    # Wait before retrying
                    wait_time = (attempt + 1) * 5
                    print(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                elif "timeout" in error_msg.lower():
                    print(f"⏰ Timeout. Retrying with smaller model...")
                    self._switch_model('down')
                    time.sleep(2)
                    
                else:
                    print(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed: {error_msg[:100]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
        
        # All retries exhausted
        return {
            'structured_output': {
                "primary_cause": f"API_ERROR: {str(last_error)[:100]}",
                "mechanism": "unknown",
                "contributing_factors": [],
                "temporal_sequence": [],
                "spatial_location": "unknown",
                "confidence": 0.0
            },
            'explanation': f"Error after {self.max_retries} attempts: {last_error}",
            'model': self.current_model,
            'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
            'structured': False,
            'error': last_error
        }
    
    def set_model(self, model_name: str):
        """Change the model after initialization."""
        self.current_model = model_name
        print(f"🔧 Model switched to: {self.current_model}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the LLM interface."""
        remaining = self.daily_limit - self.token_usage
        return {
            'model': self.current_model,
            'api_keys': len(self.api_keys),
            'current_key': self.current_key_index + 1,
            'token_usage': self.token_usage,
            'daily_limit': self.daily_limit,
            'remaining_tokens': remaining,
            'can_process_100': remaining > 100 * self._get_token_estimate()
        }


# For backward compatibility
NVIDIALLM = GroqLLM