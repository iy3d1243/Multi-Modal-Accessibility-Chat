import requests
import json

class GroqAPI:
    """
    A class to interact with Groq's API for fast language model inference.
    """

    def __init__(self):
        # Default API key - should be provided by the user
        self.api_key = ""  # User needs to provide their own API key
        # API URL
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # Default model
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def set_api_key(self, api_key):
        """Set the Groq API key"""
        if api_key and api_key.strip():
            self.api_key = api_key

    def set_model(self, model):
        """Set the model to use for inference"""
        self.model = model

    def generate_response(self, prompt):
        """Generate a response using the Groq API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 800
            }

            response = requests.post(self.api_url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"Error: API returned status code {response.status_code}. {response.text}"

        except Exception as e:
            return f"Error with Groq API: {str(e)}"

# List of available models on Groq
AVAILABLE_MODELS = [
    {
        "name": "Llama-4-Scout 17B",
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Meta's latest Llama 4 Scout model - fast and powerful"
    },
    {
        "name": "Llama-3 70B",
        "id": "meta-llama/llama-3-70b-instruct",
        "description": "Meta's largest Llama 3 model - high quality responses"
    },
    {
        "name": "Llama-3 8B",
        "id": "meta-llama/llama-3-8b-instruct",
        "description": "Smaller Llama 3 model - faster responses"
    },
    {
        "name": "Mixtral 8x7B",
        "id": "mixtral/mixtral-8x7b-instruct",
        "description": "Mixtral's powerful mixture-of-experts model"
    },
    {
        "name": "Gemma 7B",
        "id": "gemma/gemma-7b-instruct",
        "description": "Google's lightweight and efficient model"
    }
]
