import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.document_graph import create_graph
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

def create_sample_documents():
    """Create sample animal documents"""
    return [
        Document(
            page_content="alpacas are domesticated mammals valued for their soft wool and friendly demeanor.",
            metadata={
                "id": "alpaca",
                "type": "mammal",
                "number_of_legs": 4,
                "keywords": ["wool", "domesticated", "friendly"],
                "origin": "south america"
            }
        ),
        Document(
            page_content="caribou, also known as reindeer, are migratory mammals found in arctic regions.",
            metadata={
                "id": "caribou",
                "type": "mammal",
                "number_of_legs": 4,
                "keywords": ["migratory", "arctic", "herbivore", "tundra"],
                "diet": "herbivorous"
            }
        ),
        Document(
            page_content="cassowaries are flightless birds known for their colorful necks and powerful legs.",
            metadata={
                "id": "cassowary",
                "type": "bird",
                "number_of_legs": 2,
                "keywords": ["flightless", "colorful", "powerful"],
                "habitat": "rainforest"
            }
        ),
        Document(
            page_content="capybaras are the largest rodents in the world and are known for their social nature.",
            metadata={
                "id": "capybara",
                "type": "mammal",
                "number_of_legs": 4,
                "keywords": ["social", "swimming", "herbivore"],
                "habitat": "wetlands",
                "origin": "south america"
            }
        )
    ]

def setup_vector_store(documents):
    """Set up AstraDB vector store"""
    # Check for required environment variables
    required_vars = [
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_API_ENDPOINT"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Please set {var} environment variable")
    
    vector_store = AstraDBVectorStore.from_documents(
        collection_name="animals",
        documents=documents,
        embedding=OpenAIEmbeddings(),
    )
    return vector_store

def setup_retriever(vector_store):
    """Set up the GraphRetriever"""
    return GraphRetriever(
        store=vector_store,
        edges=[("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
        strategy=Eager(k=10, start_k=1, max_depth=2)
    )

def save_graph_visualization(query_results, retriever, output_path="animal_graph.png"):
    """Create and save graph visualization with document content as node labels"""
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes with document content
    for doc in query_results:
        G.add_node(doc.page_content)
    
    # Add edges based on shared metadata values
    for edge_source, edge_target in retriever.edges:
        for doc1 in query_results:
            for doc2 in query_results:
                if doc1 != doc2:
                    # Get metadata values
                    val1 = doc1.metadata.get(edge_source)
                    val2 = doc2.metadata.get(edge_target)
                    
                    # Handle both single values and lists
                    if isinstance(val1, list) and isinstance(val2, list):
                        # If both are lists, check for any common elements
                        if any(v in val2 for v in val1):
                            G.add_edge(doc1.page_content, doc2.page_content)
                    elif isinstance(val1, list):
                        if val2 in val1:
                            G.add_edge(doc1.page_content, doc2.page_content)
                    elif isinstance(val2, list):
                        if val1 in val2:
                            G.add_edge(doc1.page_content, doc2.page_content)
                    else:
                        # Direct comparison for non-list values
                        if val1 and val2 and val1 == val2:
                            G.add_edge(doc1.page_content, doc2.page_content)
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=5000,
            font_size=6,
            font_weight='bold',
            width=2,
            arrowsize=20)
    
    # Adjust layout to prevent text cutoff
    plt.margins(0.2)
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGraph visualization saved to: {output_path}")

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Set up vector store
    vector_store = setup_vector_store(documents)
    
    # Set up retriever
    retriever = setup_retriever(vector_store)
    
    # Example query
    query = "what mammals could be found near a capybara"
    query_results = retriever.invoke(query)
    
    # Print results
    print("\nQuery Results:")
    print("-------------")
    for doc in query_results:
        print(f"{doc.metadata['id']}: {doc.page_content}")
    
    # Save graph visualization
    save_graph_visualization(query_results, retriever)

if __name__ == "__main__":
    main() 