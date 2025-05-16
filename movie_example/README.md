# Movie Reviews Graph RAG Example

This example demonstrates how to use Graph RAG (Retrieval Augmented Generation) techniques to build a movie recommendation system based on movie reviews.

## Setup

1. First, ensure you have Python 3.11+ installed.

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

5. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the example:
```bash
python main.py
```

This will:
1. Load sample movie review data
2. Create a vector store with embeddings
3. Set up a graph retriever
4. Process a sample query about family movies
5. Generate movie recommendations based on the reviews

## Data

The example uses a small sample dataset of Addams Family movie reviews. You can modify the data in `main.py` to use your own movie review dataset.

## How it works

1. Movie reviews and movie information are loaded and converted to documents
2. Documents are embedded and stored in a vector store
3. A graph retriever is used to find relevant reviews and movie information
4. Results are compiled and formatted
5. An LLM generates recommendations based on the retrieved information 