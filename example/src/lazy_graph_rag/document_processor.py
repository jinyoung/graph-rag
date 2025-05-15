"""Document processor for preparing documents for the vector store."""

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import uuid
import json

class DocumentProcessor:
    """Processes documents for the vector store."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.embedding_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using OpenAI.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List[str]: List of extracted entities
        """
        prompt = f"""
        다음 텍스트에서 중요한 개체(인물, 조직, 장소, 개념 등)를 추출해주세요.
        결과는 JSON 배열 형식으로 반환해주세요.
        중요: 반드시 유효한 JSON 배열 형식으로만 응답해주세요.
        
        텍스트: {text}
        
        예시 형식:
        ["서울", "대한민국", "AI"]
        """
        
        response = self.llm.invoke(prompt)
        try:
            # Clean the response to ensure it only contains the JSON array
            content = response.content.strip()
            if content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()
            entities = json.loads(content)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Warning: Failed to extract entities from text: {e}")
            return []

    def prepare_batch(self, chunks: List[str]) -> List[Document]:
        """Prepare a batch of documents from chunks.
        
        Args:
            chunks: List of text chunks to process
            
        Returns:
            List[Document]: Processed documents
        """
        documents = []
        for chunk in chunks:
            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid4())
            
            # Extract entities first
            entities = self.extract_entities(chunk)
            
            # Generate embedding
            embedding = self.embedding_model.embed_query(chunk)
            
            # Create document with metadata and ID
            doc = Document(
                id=chunk_id,
                page_content=chunk,
                metadata={
                    'chunk_id': chunk_id,
                    'entities': entities,
                    'embedding': embedding
                }
            )
            
            documents.append(doc)
            
        return documents 