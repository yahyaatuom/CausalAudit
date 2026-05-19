import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Optional: uncomment if you want retry logic
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Force load from the correct path
env_path = r"C:\Users\Dell\Desktop\Causal-Guard\.env"
load_dotenv(env_path)


class GroqLLM:
    def __init__(self, model="deepseek-ai/deepseek-r1-distill-qwen-7b"):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = model
        self.temperature = 0
        self.max_retries = 2  # Add retry count
        
        print(f"DEBUG: API Key loaded: {'Yes' if self.api_key else 'No'}")
        print(f"DEBUG: Using model: {self.model}")
        
        if not self.api_key:
            print("⚠️ Error: NVIDIA_API_KEY not found. Check your .env file.")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
    
    def _build_prompt(self, scenario_description):
        """Build the prompt (separate method for easier modification)"""
        return f"""Analyze the following incident and provide a concise explanation focusing on the scenario.

Incident: {scenario_description}

Explanation:"""
    
    def generate_explanation(self, scenario_description):
        """
        Send scenario to NVIDIA NIM and get explanation with retry logic
        """
        if not self.client:
            return {
                'explanation': "Error: API key not configured",
                'model': self.model,
                'tokens': {'prompt': 0, 'completion': 0, 'total': 0}
            }
        
        prompt = self._build_prompt(scenario_description)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500,
                    timeout=45  # Add timeout
                )
                
                return {
                    'explanation': response.choices[0].message.content,
                    'model': self.model,
                    'tokens': {
                        'prompt': response.usage.prompt_tokens,
                        'completion': response.usage.completion_tokens,
                        'total': response.usage.total_tokens
                    }
                }
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        'explanation': f"Error after {self.max_retries} attempts: {str(e)}",
                        'model': self.model,
                        'tokens': {'prompt': 0, 'completion': 0, 'total': 0}
                    }
                time.sleep(2)  # Wait before retry
        
        # Fallback (should not reach here)
        return {
            'explanation': "Unexpected error occurred",
            'model': self.model,
            'tokens': {'prompt': 0, 'completion': 0, 'total': 0}
        }
    
    def set_model(self, model_name):
        """Change the model right after initialization"""
        self.model = model_name
        print(f"✅ Model changed to: {self.model}")