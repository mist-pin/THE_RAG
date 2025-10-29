"""
Vector DB Client: Handles all interactions with external Vector Database
"""

from typing import List


class VectorDBClient:
    """Client for Vector Database operations"""
    
    def __init__(self, api_key: str = None, endpoint: str = None):
        """
        Initialize Vector DB client
        
        Args:
            api_key: API key for vector DB service
            endpoint: Vector DB endpoint URL
        """
        self.api_key = api_key
        self.endpoint = endpoint
    
    def search(self, features: List[str], top_k: int = 5) -> List[str]:
        """
        Search vector DB using features and return relevant chunks
        
        Args:
            features: List of extracted features to search for
            top_k: Number of top results to return
            
        Returns:
            List of relevant text chunks from vector DB
        """
        mock_chunks = [
            f"Mr. Shashank made purchases on 18-08-2023 including laptop ($1200) and mouse ($25).",
            f"Transaction details: Customer: Shashank, Date: 18-08-2023, Total: $1225",
            f"Purchase history shows Mr. Shashank is a regular customer since 2022."
        ]

        
        
        return mock_chunks
    
    def embed_and_store(self, text: str, metadata: dict = None) -> bool:
        """
        Embed text and store in vector DB
        
        Args:
            text: Text to embed and store
            metadata: Additional metadata for the chunk
            
        Returns:
            Success status
        """
        pass
