"""Document processing and entity extraction module."""

from typing import List, Iterator
from uuid import uuid4

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

    def parse_document(self, text: str, chunk_id: str = None) -> Document:
        """Creates a Document from a text chunk.
        
        Args:
            text: A text chunk to process
            chunk_id: Optional ID for the chunk. If not provided, a UUID will be generated.
            
        Returns:
            Document: A LangChain Document
        """
        if chunk_id is None:
            chunk_id = str(uuid4())

        return Document(
            id=chunk_id,
            page_content=text,
            metadata={
                "chunk_id": chunk_id,
            },
        )

    def prepare_batch(self, chunks: List[str]) -> Iterator[Document]:
        """Process a batch of text chunks and extract entities.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Iterator[Document]: Processed documents with extracted entities
        """
        # Parse documents from the chunks
        docs = [self.parse_document(chunk) for chunk in chunks]

        # Extract entities using spaCy
        docs = self.ner_transformer.transform_documents(docs)

        return docs 