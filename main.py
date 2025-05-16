import os
from dotenv import load_dotenv
from langchain.vectorstores import OpenAIEmbeddings

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")
    
    # Initialize embedding model with API key
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key) 