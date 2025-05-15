import requests
import json
import os
from huggingface_hub import InferenceClient

class HuggingFaceAPI:
    """
    A class to interact with Hugging Face's free inference API for text generation.
    This provides a free alternative to OpenAI's API.
    """
    
    def __init__(self):
        # Default API token - users can set their own
        self.api_token = ""
        # Default model - a good free alternative that doesn't require API token
        self.model = "google/flan-t5-large"
        # Inference API URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        # Initialize the client as None until token is set
        self.client = None
        
    def set_api_token(self, api_token):
        """Set the Hugging Face API token"""
        self.api_token = api_token
        # Initialize the client with the token
        if api_token:
            self.client = InferenceClient(api_token)
        
    def set_model(self, model):
        """Set the model to use for inference"""
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        
    def generate_response(self, prompt):
        """Generate a response using the Hugging Face Inference API"""
        # If using client with API token
        if self.client and self.api_token:
            try:
                # Use the client for inference
                response = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=150,
                    temperature=0.7,
                    repetition_penalty=1.2
                )
                return response
            except Exception as e:
                return f"Error with Hugging Face API: {str(e)}"
        
        # Fallback to direct API call without token (limited usage)
        try:
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
                
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "repetition_penalty": 1.2,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        return result[0]["generated_text"]
                    else:
                        return str(result[0])
                else:
                    return str(result)
            else:
                return f"Error: API returned status code {response.status_code}. {response.text}"
                
        except Exception as e:
            return f"Error with Hugging Face API: {str(e)}"

# List of free models that work well for chat
AVAILABLE_MODELS = [
    {
        "name": "Google FLAN-T5-Large",
        "id": "google/flan-t5-large",
        "description": "A good general-purpose model for instruction following"
    },
    {
        "name": "Mistral 7B Instruct",
        "id": "mistralai/Mistral-7B-Instruct-v0.1",
        "description": "Powerful instruction-following model (requires API token)"
    },
    {
        "name": "TinyLlama",
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Small but effective chat model"
    },
    {
        "name": "Falcon 7B Instruct",
        "id": "tiiuae/falcon-7b-instruct",
        "description": "Good for general chat (requires API token)"
    }
]
