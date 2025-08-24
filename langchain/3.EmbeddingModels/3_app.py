import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load .env and initialize client
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

client = InferenceClient(token=hf_token)

# Helper functions
def embed_text(texts, model="sentence-transformers/all-MiniLM-L6-v2"):
    """Get embeddings for list of texts"""
    embeddings = []
    for t in texts:
        emb = client.feature_extraction(t, model=model)
        embeddings.append(np.array(emb))
    return np.vstack(embeddings)

def semantic_search(query, documents, doc_embeddings):
    """Return the single most similar doc as one-sentence answer"""
    query_emb = embed_text([query])
    sims = cosine_similarity(query_emb, doc_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return documents[best_idx], float(sims[best_idx])

# Pokémon documents (one sentence each)
documents = [
    "Pikachu is an Electric-type Pokémon known for its powerful Thunderbolt attack.",
    "Charizard is a Fire and Flying type Pokémon that can mega evolve into two forms.",
    "Bulbasaur is a Grass and Poison type Pokémon, famous for its move Vine Whip.",
    "Mewtwo is a Legendary Psychic-type Pokémon created from genetic experiments.",
    "Gengar is a Ghost and Poison type Pokémon that loves to hide in shadows.",
    "Squirtle is a Water-type Pokémon that can evolve into Wartortle and then Blastoise.",
    "Eevee is a unique Pokémon that can evolve into many different elemental forms.",
]

# Pre-compute embeddings once
doc_embeddings = embed_text(documents)

# Streamlit UI
st.title("🔍 Pokémon Semantic Search")
st.write("Ask a question about the Pokémon listed below. The app will return the closest one-sentence answer.")

# Show documents so user knows the scope
st.subheader("📜 Knowledge Base")
for i, doc in enumerate(documents, start=1):
    st.write(f"**{i}. {doc}**")

# Query input
st.subheader("🔎 Search")
query = st.text_input("Enter your question:")

if query:
    answer, score = semantic_search(query, documents, doc_embeddings)
    st.subheader("Answer:")
    st.write(f"**{answer}** (similarity: {score:.4f})")
