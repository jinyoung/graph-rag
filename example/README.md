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
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
LANGCHAIN_API_KEY=your_langchain_key  # Optional
```

4. Set up Supabase pgvector:
- Create a new Supabase project
- Enable the pgvector extension in your Supabase database
- Create a new table for vectors with the following SQL:
```sql
create table lazy_graph_rag_korean (
  id uuid primary key,
  content text,
  metadata jsonb,
  embedding vector(1536)
);

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (lazy_graph_rag_korean.embedding <=> query_embedding) as similarity
  from lazy_graph_rag_korean
  order by lazy_graph_rag_korean.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

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