import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": "Bearer YOUR_API_TOKEN"}  # Replace with your actual token when using

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Example input
user_input = "Hi there! How are you today?"

# Uncomment to run the query
# response = query({"inputs": user_input})
# print("🤖 Bot reply:", response)

print("This is a test file. Replace YOUR_API_TOKEN with your actual token to use it.")
