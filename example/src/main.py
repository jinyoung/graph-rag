"""Main application entry point."""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from graph_rag_example_helpers.datasets.wikimultihop import aload_2wikimultihop
from langchain_astradb import AstraDBVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_graph_retriever import GraphRetriever

from lazy_graph_rag.document_processor import DocumentProcessor
from lazy_graph_rag.graph_rag import LazyGraphRAG

# Load environment variables
load_dotenv()

# Configuration
USE_SHORT_DATASET = True
COLLECTION = "lazy_graph_rag_short" if USE_SHORT_DATASET else "lazy_graph_rag"
PARA_WITH_HYPERLINK_ZIP = os.path.join(os.path.dirname(__file__), "../data/para_with_hyperlink.zip")

async def setup_vector_store() -> AstraDBVectorStore:
    """Set up and populate the vector store.
    
    Returns:
        AstraDBVectorStore: Configured vector store
    """
    # Initialize vector store
    store = AstraDBVectorStore(
        embedding=OpenAIEmbeddings(),
        collection_name=COLLECTION,
        pre_delete_collection=USE_SHORT_DATASET,
    )

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Load data into store
    await aload_2wikimultihop(
        limit=100 if USE_SHORT_DATASET else None,
        full_para_with_hyperlink_zip_path=PARA_WITH_HYPERLINK_ZIP,
        store=store,
        batch_prepare=doc_processor.prepare_batch,
    )

    return store

async def setup_lazy_graph_rag(store: AstraDBVectorStore) -> LazyGraphRAG:
    """Set up LazyGraphRAG with the vector store.
    
    Args:
        store: Vector store to use
        
    Returns:
        LazyGraphRAG: Configured LazyGraphRAG instance
    """
    # Initialize retriever
    retriever = GraphRetriever(
        store=store,
        edges=[("mentions", "$id"), ("entities", "entities")],
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
    store = await setup_vector_store()

    # Set up LazyGraphRAG
    print("Setting up LazyGraphRAG...")
    lazy_graph_rag = await setup_lazy_graph_rag(store)

    # Example questions
    questions = [
        "Why are Bermudan sloop ships widely prized compared to other ships?",
        "What is the relationship between the Nile River and ancient Egyptian civilization?",
        "How did the invention of the printing press impact medieval Europe?",
    ]

    # Process questions
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = await lazy_graph_rag.ainvoke(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 