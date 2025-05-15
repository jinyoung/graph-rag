"""Main application entry point."""

import asyncio
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_graph_retriever import GraphRetriever
from supabase.client import create_client
import os

from lazy_graph_rag.document_processor import DocumentProcessor
from lazy_graph_rag.graph_rag import LazyGraphRAG
from lazy_graph_rag.chunks import CHUNKS
from adapters.supabase import SupabaseVectorStore, SupabaseAdapter

# Load environment variables
load_dotenv()

# Configuration
TABLE_NAME = "lazy_graph_rag_korean"
QUERY_NAME = "match_documents"

async def setup_vector_store() -> SupabaseAdapter:
    """Set up and populate the vector store.
    
    Returns:
        SupabaseAdapter: Configured vector store adapter
    """
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize vector store
    store = SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(),
        table_name=TABLE_NAME,
        query_name=QUERY_NAME
    )

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Process chunks and add to store
    docs = doc_processor.prepare_batch(CHUNKS)
    await store.aadd_documents(docs)

    # Create and return the adapter
    return SupabaseAdapter(store)

async def setup_lazy_graph_rag(adapter: SupabaseAdapter) -> LazyGraphRAG:
    """Set up LazyGraphRAG.
    
    Args:
        adapter: Vector store adapter to use
        
    Returns:
        LazyGraphRAG: Configured LazyGraphRAG instance
    """
    # Initialize components
    retriever = GraphRetriever(
        store=adapter,
        edges=[("entities", "entities")],
        k=100,
        start_k=30,
        adjacent_k=20,
        max_depth=3,
    )

    # Initialize model
    model = ChatOpenAI(model="gpt-4", temperature=0)

    # Create LazyGraphRAG instance
    return LazyGraphRAG(retriever=retriever, model=model)

async def main() -> None:
    """Main application entry point."""
    # Set up vector store
    print("Setting up vector store...")
    adapter = await setup_vector_store()

    # Set up LazyGraphRAG
    print("Setting up LazyGraphRAG...")
    lazy_graph_rag = await setup_lazy_graph_rag(adapter)

    # Example questions
    questions = [
        "영희와 철수의 관계는 어떤가요?",
        "수진이는 어디에 살고 있나요?",
        "부산에 대해 알려진 정보는 무엇인가요?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        # Get documents from the retriever
        docs = await lazy_graph_rag.retriever.aget_relevant_documents(question)
        
        # Visualize the document graph
        lazy_graph_rag.visualize_graph(docs, save_path=f"graph_{question.replace(' ', '_')}.png")
        
        # Get the answer
        answer = await lazy_graph_rag.ainvoke(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 