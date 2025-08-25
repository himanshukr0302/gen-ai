import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key was found
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")

# Configure the Gemini API with the API key
genai.configure(api_key=api_key)

def list_gemini_models():
    """
    Fetches and prints all available Gemini models that support text generation.
    """
    print("Available Gemini models for text generation:")
    print("------------------------------------------")

    try:
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                # Print the model's name and a short description
                print(f"Name: {model.name}")
                print(f"  Description: {model.description}")
                print(f"  Input Token Limit: {model.input_token_limit}")
                print(f"  Output Token Limit: {model.output_token_limit}")
                print("-" * 42)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_gemini_models()