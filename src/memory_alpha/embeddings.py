"""
Embeddings module for Memory Alpha.

This module provides functions to generate embeddings using Ollama.
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from memory_alpha.settings import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Client for generating embeddings using Ollama API."""
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize the Ollama embedder.
        
        Args:
            model: The Ollama model to use for embeddings (default: from settings)
            base_url: The base URL of the Ollama API (default: from settings)
        """
        self.model = model or settings.embed_model
        self.base_url = base_url or settings.ollama_url
        self.embed_endpoint = f"{self.base_url.rstrip('/')}/api/embeddings"
        
        # Check if the model exists
        try:
            resp = requests.get(f"{self.base_url.rstrip('/')}/api/tags")
            available_models = [model["name"] for model in resp.json()["models"]]
            if self.model not in available_models:
                logger.warning(
                    f"Model {self.model} not found in Ollama. "
                    f"Available models: {', '.join(available_models)}. "
                    f"You may need to run 'ollama pull {self.model}'"
                )
        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        payload = {
            "model": self.model,
            "prompt": text,
        }
        
        try:
            response = requests.post(
                self.embed_endpoint,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            
            # Normalize the embedding (important for high-quality vector search)
            if embedding:
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding = (embedding_array / norm).tolist()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding with Ollama: {e}")
            # Return a zero vector of the expected dimension as fallback
            return [0.0] * settings.embed_dim
    
    def bulk_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return [self.embed(text) for text in texts]


# Singleton instance for use throughout the application
embedder = OllamaEmbedder()


def embed_text(text: str) -> List[float]:
    """Generate embeddings for a text string.
    
    This is the main function that should be used by other modules.
    
    Args:
        text: The text to embed
        
    Returns:
        List[float]: The embedding vector
    """
    return embedder.embed(text)