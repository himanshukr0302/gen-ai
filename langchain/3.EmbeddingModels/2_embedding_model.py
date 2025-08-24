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

# List of sentences/documents
documents = [
    "Today is a sunny day and I will get some ice cream.",
    "Machine learning is a subset of artificial intelligence.",
    "I love programming in Python.",
    "The capital of India is New Delhi."
]

# Generate embeddings for each document
embeddings = []
for doc in documents:
    emb = client.feature_extraction(
        doc,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings.append(emb)

# Print embeddings
for i, emb in enumerate(embeddings, start=1):
    print(f"Document {i} embedding (length {len(emb)}):\n{emb[:10]}...\n")  # printing first 10 dims
