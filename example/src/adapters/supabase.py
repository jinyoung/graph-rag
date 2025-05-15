"""Provides an adapter for Supabase with pgvector integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
import uuid

import backoff
from graph_retriever import Content
from graph_retriever.edges import Edge, IdEdge, MetadataEdge
from graph_retriever.utils import merge
from graph_retriever.utils.batched import batched
from graph_retriever.utils.top_k import top_k
from typing_extensions import override

from langchain_core.documents import Document
from graph_retriever.adapters import Adapter

_EXCEPTIONS_TO_RETRY = (
    Exception,  # Replace with specific Supabase exceptions if needed
)
_MAX_RETRIES = 3


def _extract_queries(edges: set[Edge]) -> tuple[dict[str, Any], set[str]]:
    metadata: dict[str, set[Any]] = {}
    ids: set[str] = set()

    for edge in edges:
        if isinstance(edge, MetadataEdge):
            metadata.setdefault(edge.incoming_field, set()).add(edge.value)
        elif isinstance(edge, IdEdge):
            ids.add(edge.id)
        else:
            raise ValueError(f"Unsupported edge {edge}")

    return (cast(dict[str, Any], metadata), ids)


class SupabaseVectorStore:
    """
    Vector store implementation for Supabase with pgvector integration.
    """

    def __init__(
        self,
        client,
        embedding,
        table_name: str = "documents",
        query_name: str = "match_documents"
    ) -> None:
        self.client = client
        self.embedding_function = embedding
        self.table_name = table_name
        self.query_name = query_name

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Return documents that are most similar to the given embedding."""
        # Convert filter to Supabase format if needed
        filter_dict = {} if filter is None else filter
        
        # Only pass the required parameters to match_documents function
        response = self.client.rpc(
            self.query_name,
            {
                'query_embedding': embedding,
                'match_count': k
            }
        ).execute()
        
        documents = []
        for match in response.data:
            metadata = match.get('metadata', {})
            metadata['score'] = match.get('similarity')
            doc = Document(
                id=match.get('id'),  # Set document ID from the database record
                page_content=match.get('content'),
                metadata=metadata
            )
            documents.append((doc, match.get('similarity')))
            
        return documents

    def get_matches_by_ids(
        self,
        ids: list[str],
        k: int = 4,
        filter: dict | None = None
    ) -> list[Document]:
        """Return documents by their IDs."""
        # Convert filter to Supabase format if needed
        filter_dict = {} if filter is None else filter
        
        response = self.client.table(self.table_name)\
            .select('*')\
            .in_('id', ids)\
            .execute()
            
        documents = []
        for match in response.data[:k]:
            metadata = match.get('metadata', {})
            doc = Document(
                id=match.get('id'),  # Set document ID from the database record
                page_content=match.get('content'),
                metadata=metadata
            )
            documents.append(doc)
            
        return documents

    async def aadd_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        for doc in documents:
            # Use chunk_id as the document id
            doc_id = doc.metadata.get('chunk_id')
            if not doc_id:
                raise ValueError("Document must have a chunk_id in its metadata")
            
            # Add the document to the vector store
            self.client.table(self.table_name)\
                .insert({
                    'id': doc_id,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'embedding': doc.metadata.get('embedding')
                })\
                .execute()


class SupabaseAdapter(Adapter):
    """
    Adapter for Supabase with pgvector integration.

    This class integrates the Supabase vector store with the graph
    retriever system, providing functionality for similarity search and document
    retrieval using pgvector.

    Parameters
    ----------
    vector_store :
        The Supabase vector store instance.
    """

    def __init__(self, vector_store: SupabaseVectorStore) -> None:
        self.vector_store = vector_store

    def _build_content(self, doc: Document) -> Content:
        return Content(
            id=doc.id,
            content=doc.page_content,
            metadata=doc.metadata,
            embedding=doc.metadata.get("embedding"),
        )

    def _build_content_list(self, docs: list[Document]) -> list[Content]:
        return [self._build_content(doc) for doc in docs]

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def _run_query(
        self,
        *,
        embedding: list[float] | None = None,
        k: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        if k == 0:
            return []

        if ids:
            # Handle ID-based query
            results = self.vector_store.get_matches_by_ids(
                ids=ids,
                k=k,
                filter=filter,
            )
        elif embedding:
            # Handle embedding-based similarity search
            results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                embedding=embedding,
                k=k,
                filter=filter,
            )
            results = [doc for doc, _ in results]  # Remove scores
        else:
            return []

        return results

    @override
    def search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding = self.vector_store.embedding_function.embed_query(query)
        docs = self._run_query(embedding=query_embedding, k=k, filter=filter)
        return query_embedding, self._build_content_list(docs)

    @override
    def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        if k == 0:
            return []
        docs = self._run_query(embedding=embedding, k=k, filter=filter)
        return self._build_content_list(docs)

    @override
    def get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        docs = self._run_query(k=len(ids), ids=list(ids), filter=filter)
        return self._build_content_list(docs)

    @override
    def adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> list[Content]:
        metadata, ids = _extract_queries(edges)
        
        results: dict[str, Content] = {}
        
        # Handle metadata-based queries
        if metadata:
            metadata_filter = {**filter} if filter else {}
            for field, values in metadata.items():
                metadata_filter[field] = {"$in": list(values)}
            docs = self._run_query(
                embedding=query_embedding,
                k=k,
                filter=metadata_filter
            )
            for doc in docs:
                results[doc.id] = self._build_content(doc)

        # Handle ID-based queries
        if ids:
            for id_batch in batched(ids, 100):
                docs = self._run_query(k=k, ids=list(id_batch), filter=filter)
                for doc in docs:
                    results[doc.id] = self._build_content(doc)

        return list(top_k(results.values(), embedding=query_embedding, k=k))

    # Async methods - These could be implemented similarly to sync methods
    # but using Supabase's async client if available

    @override
    async def asearch_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        # For now, use sync version
        return self.search_with_embedding(query, k, filter, **kwargs)

    @override
    async def asearch(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        # For now, use sync version
        return self.search(embedding, k, filter, **kwargs)

    @override
    async def aget(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        # For now, use sync version
        return self.get(ids, filter, **kwargs)

    @override
    async def aadjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> list[Content]:
        # For now, use sync version
        return self.adjacent(edges, query_embedding, k, filter, **kwargs) 