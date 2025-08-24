import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Check your .env file.")

# Initialize client
client = InferenceClient(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    token=hf_token
)

# Test call
resp = client.chat_completion(
    messages=[{"role": "user", "content": "Explain the concept of recursion in programming."}],
    max_tokens=250
)

print(resp.choices[0].message["content"])
