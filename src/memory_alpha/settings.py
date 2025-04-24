from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default context levels
DEFAULT_CONTEXT_LEVELS = ["function_signature", "file_section", "file", "module", "project"]


class Settings(BaseSettings):
    """Application settings from environment variables with defaults.
    
    Environment variables mapping:
    - QDRANT_URL: URL of the Qdrant vector database
    - OPENAI_API_KEY: OpenAI API key
    - EMBED_MODEL: Name of the embedding model
    - EMBED_DIM: Dimension of the embeddings
    - CLUSTER_COLLECTION: Name of the cluster collection
    - CHUNK_COLLECTION: Name of the chunk collection
    - DEFAULT_MAX_TOKENS: Maximum number of tokens to return
    - DEFAULT_K: Number of chunks to retrieve per level
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    # Server settings
    qdrant_url: str = "http://localhost:6333"
    openai_api_key: Optional[str] = None
    
    # Model settings
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    
    # Collection names
    cluster_collection: str = "dev_clusters"
    chunk_collection: str = "dev_chunks"
    
    # Default parameters
    default_max_tokens: int = 1000
    default_k: int = 24

settings = Settings()
