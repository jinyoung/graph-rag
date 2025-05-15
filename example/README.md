# LazyGraphRAG Example

This is a Python implementation of the LazyGraphRAG system, which demonstrates significant cost and performance benefits by delaying the construction of a knowledge graph until query time.

## Setup

1. Install dependencies using uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate  # On Windows
uv pip install -r requirements.txt
```

2. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_key
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
LANGCHAIN_API_KEY=your_langchain_key  # Optional
ASTRA_DB_KEYSPACE=your_keyspace  # Optional
```

4. Download the Wikipedia dataset:
- Download `para_with_hyperlink.zip` from [2wikimultihop](https://github.com/Alab-NII/2wikimultihop)
- Place it in the `data` directory

## Usage

Run the example:
```bash
python src/main.py
```

## Project Structure

- `src/`: Source code
  - `main.py`: Main application entry point
  - `lazy_graph_rag/`: Core implementation
    - `document_processor.py`: Document processing and entity extraction
    - `graph_rag.py`: LazyGraphRAG implementation
    - `claims.py`: Claims extraction and ranking
- `data/`: Data files
- `tests/`: Test files

## Features

- Document graph construction with Wikipedia articles
- Entity extraction using spaCy
- Claims extraction from document communities
- Claim ranking based on relevance
- Question answering using selected claims 