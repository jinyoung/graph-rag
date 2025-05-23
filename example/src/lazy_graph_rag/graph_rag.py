"""LazyGraphRAG implementation."""

from collections.abc import Iterable
from typing import Any, Optional, Union, List

from graph_retriever.edges import EdgeSpec, MetadataEdgeFunction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.document_graph import create_graph, group_by_community
from langchain_core.documents import Document
import networkx as nx
import matplotlib.pyplot as plt

from .claims import Claim, ClaimsProcessor

class LazyGraphRAG:
    """LazyGraphRAG implementation."""

    def __init__(
        self,
        retriever: GraphRetriever,
        model: BaseLanguageModel,
        edges: Optional[Union[Iterable[EdgeSpec], MetadataEdgeFunction]] = None,
        max_tokens: int = 1000,
    ) -> None:
        """Initialize LazyGraphRAG.
        
        Args:
            retriever: Graph retriever for document retrieval
            model: Language model for processing
            edges: Edge specifications for graph construction
            max_tokens: Maximum tokens for claims
        """
        self.retriever = retriever
        self.model = model
        self.edges = edges or retriever.edges
        self.max_tokens = max_tokens
        self.claims_processor = ClaimsProcessor(model)

        if self.edges is None:
            raise ValueError("Must specify 'edges' in constructor or retriever")

        # Initialize answer prompt
        self.answer_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the supporting claims.

Only use information from the claims. Do not guess or make up any information.

Where possible, reference and quote the supporting claims.

Question: {question}

Claims:
{claims}
""")

        # Build the chain
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "claims": RunnablePassthrough()
                | self._process_question,
            }
            | self.answer_prompt
            | model
        )

    async def _process_question(self, question: str) -> str:
        """Process a question through the LazyGraphRAG pipeline.
        
        Args:
            question: Question to process
            
        Returns:
            str: Formatted claims string
        """
        # 1. Retrieve documents using the graph retriever
        documents = await self.retriever.ainvoke(question, edges=self.edges)

        # 2. Create graph and extract communities
        document_graph = create_graph(documents, edges=self.edges)
        communities = group_by_community(document_graph)

        # 3. Extract claims from communities
        claims = await self.claims_processor.extract_claims(question, communities)

        # 4. Rank claims and select up to token limit
        result_claims = []
        tokens = 0

        for claim in await self.claims_processor.rank_claims(question, claims):
            claim_str = f"- {claim.claim} (Source: {claim.source_id})"

            tokens += self.model.get_num_tokens(claim_str)
            if tokens > self.max_tokens:
                break
            result_claims.append(claim_str)

        return "\n".join(result_claims)

    async def ainvoke(self, question: str, **kwargs: Any) -> str:
        """Answer a question using LazyGraphRAG.
        
        Args:
            question: Question to answer
            **kwargs: Additional arguments
            
        Returns:
            str: Answer to the question
        """
        result = await self.chain.ainvoke(question)
        return result.content

    def visualize_graph(self, documents: List[Document], save_path: Optional[str] = None):
        """
        Visualize the document graph using NetworkX and matplotlib.
        
        Args:
            documents: List of documents to create the graph from
            save_path: Optional path to save the graph visualization. If None, displays the graph.
        """
        document_graph = create_graph(
            documents=documents,
            edges=self.edges,
        )
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(document_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(document_graph, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(document_graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: f"Doc {node[:8]}..." for node in document_graph.nodes()}
        nx.draw_networkx_labels(document_graph, pos, labels, font_size=8)
        
        plt.title("Document Graph Visualization")
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close() 