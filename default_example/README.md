# Animal Graph RAG Example

This is a basic example demonstrating how to combine Graph Traversal and Vector Search using `langchain-graph-retriever` with `langchain`. The example uses a simple dataset about animals to show how to:

1. Create and structure documents with metadata
2. Set up a vector store (AstraDB)
3. Configure graph traversal
4. Visualize the document relationships

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in a `.env` file:
```
# OpenAI API Key for embeddings
OPENAI_API_KEY=your-api-key-here

# AstraDB Configuration
ASTRA_DB_APPLICATION_TOKEN=your-astra-token-here
ASTRA_DB_API_ENDPOINT=your-astra-api-endpoint-here
```

### Setting up AstraDB

1. Create an account at [DataStax](https://astra.datastax.com)
2. Create a new Vector Database
3. Get your API endpoint and application token from the database settings
4. Add these to your `.env` file as shown above

## Running the Example

Simply run:
```bash
python main.py
```

This will:
1. Create sample animal documents
2. Set up an AstraDB vector store
3. Configure graph traversal with habitat, origin, and keyword connections
4. Run a sample query about mammals near capybaras
5. Generate a visualization of the document relationships

## Output

The script will:
1. Print the query results showing related animals
2. Save a graph visualization as `animal_graph.png`

## Understanding the Code

The example demonstrates:
- Document creation with structured metadata
- Vector store setup with AstraDB
- Graph retriever configuration with multiple edge types
- Simple graph visualization using networkx

Check the comments in `main.py` for detailed explanations of each step. 