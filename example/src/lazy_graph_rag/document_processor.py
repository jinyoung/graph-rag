"""Document processing and entity extraction module."""

import json
from collections.abc import Iterator
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_graph_retriever.transformers.spacy import SpacyNERTransformer

class DocumentProcessor:
    """Processes documents and extracts entities."""
    
    def __init__(self) -> None:
        """Initialize the document processor."""
        self.ner_transformer = SpacyNERTransformer(
            limit=1000,
            exclude_labels={"CARDINAL", "MONEY", "QUANTITY", "TIME", "PERCENT", "ORDINAL"},
        )

    def parse_document(self, line: bytes) -> Document:
        """Reads one JSON line from the wikimultihop dump.
        
        Args:
            line: A JSON line from the wikimultihop dataset
            
        Returns:
            Document: A LangChain Document with extracted metadata
        """
        para = json.loads(line)

        id = para["id"]
        title = para["title"]

        # Use structured information (mentioned Wikipedia IDs) as metadata
        mentioned_ids = [id for m in para["mentions"] for m in m["ref_ids"] or []]

        return Document(
            id=id,
            page_content=" ".join(para["sentences"]),
            metadata={
                "mentions": mentioned_ids,
                "title": title,
            },
        )

    def prepare_batch(self, lines: Iterator[str]) -> Iterator[Document]:
        """Process a batch of documents and extract entities.
        
        Args:
            lines: Iterator of JSON lines
            
        Returns:
            Iterator[Document]: Processed documents with extracted entities
        """
        # Parse documents from the batch of lines
        docs = [self.parse_document(line) for line in lines]

        # Extract entities using spaCy
        docs = self.ner_transformer.transform_documents(docs)

        return docs 