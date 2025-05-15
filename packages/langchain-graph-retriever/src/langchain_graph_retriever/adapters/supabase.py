"""Provides an adapter for Supabase with pgvector integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import backoff
from graph_retriever import Content
from graph_retriever.edges import Edge, IdEdge, MetadataEdge
from graph_retriever.utils import merge
from graph_retriever.utils.batched import batched
from graph_retriever.utils.top_k import top_k
from typing_extensions import override

try:
    from langchain_community.vectorstores.supabase import SupabaseVectorStore
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-community supabase-py`")

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


class SupabaseAdapter(Adapter):
    """
    Adapter for Supabase with pgvector integration.

    This class integrates the LangChain Supabase vector store with the graph
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