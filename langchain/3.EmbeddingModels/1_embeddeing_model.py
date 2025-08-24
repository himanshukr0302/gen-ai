import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Get HF token from .env
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please add it to your .env file.")

# Initialize client with token
client = InferenceClient(token=hf_token)

# Run feature extraction
result = client.feature_extraction(
    "Today is a sunny day and I will get some ice cream.",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print(result)
