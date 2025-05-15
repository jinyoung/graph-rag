"""Main application entry point."""

import asyncio
from typing import Optional

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_graph_retriever import GraphRetriever

from lazy_graph_rag.document_processor import DocumentProcessor
from lazy_graph_rag.graph_rag import LazyGraphRAG
from lazy_graph_rag.chunks import CHUNKS

# Load environment variables
load_dotenv()

# Configuration
COLLECTION = "lazy_graph_rag_korean"

async def setup_vector_store() -> AstraDBVectorStore:
    """Set up and populate the vector store.
    
    Returns:
        AstraDBVectorStore: Configured vector store
    """
    # Initialize vector store
    store = AstraDBVectorStore(
        embedding=OpenAIEmbeddings(),
        collection_name=COLLECTION,
        pre_delete_collection=True,  # Always start fresh for our example
    )

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Process chunks and add to store
    docs = doc_processor.prepare_batch(CHUNKS)
    await store.aadd_documents(docs)

    return store

async def setup_lazy_graph_rag(store: AstraDBVectorStore) -> LazyGraphRAG:
    """Set up LazyGraphRAG.
    
    Args:
        store: Vector store to use
        
    Returns:
        LazyGraphRAG: Configured LazyGraphRAG instance
    """
    # Initialize components
    retriever = GraphRetriever(
        store=store,
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
    store = await setup_vector_store()

    # Set up LazyGraphRAG
    print("Setting up LazyGraphRAG...")
    lazy_graph_rag = await setup_lazy_graph_rag(store)

    # Example questions
    questions = [
        "영희와 철수의 관계는 어떤가요?",
        "수진이는 어디에 살고 있나요?",
        "부산에 대해 알려진 정보는 무엇인가요?",
    ]

    # Process questions
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = await lazy_graph_rag.ainvoke(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 