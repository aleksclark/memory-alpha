from typing import List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default context levels
DEFAULT_CONTEXT_LEVELS = [
    "function_signature",
    "file_section",
    "file",
    "module",
    "project",
]

# Server mode types
ServerMode = Literal["stdio", "sse"]


class Settings(BaseSettings):
    """Application settings from environment variables with defaults.

    Environment variables mapping:
    - QDRANT_URL: URL of the Qdrant vector database
    - OLLAMA_URL: URL of the Ollama server (default: http://localhost:11434)
    - EMBED_MODEL: Name of the embedding model
    - EMBED_DIM: Dimension of the embeddings
    - COLLECTION_PREFIX: Prefix for collection names
    - CLUSTER_COLLECTION: Name of the cluster collection (derived from prefix)
    - CHUNK_COLLECTION: Name of the chunk collection (derived from prefix)
    - DEFAULT_MAX_TOKENS: Maximum number of tokens to return
    - DEFAULT_K: Number of chunks to retrieve per level
    - SERVER_MODE: Mode to run server in (stdio or sse)
    - SERVER_HOST: Host to bind to when in SSE mode
    - SERVER_PORT: Port to listen on when in SSE mode
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
    ollama_url: str = "http://localhost:11434"
    server_mode: ServerMode = "stdio"
    server_host: str = "0.0.0.0"
    server_port: int = 8080

    # Model settings
    embed_model: str = "mxbai-embed-large:latest"
    embed_dim: int = 1024  # mxbai-embed-large has 1024 dimensions

    # Collection names
    collection_prefix: str = "dev_"
    
    @property
    def cluster_collection(self) -> str:
        return f"{self.collection_prefix}clusters"
    
    @property
    def chunk_collection(self) -> str:
        return f"{self.collection_prefix}chunks"

    # Default parameters
    default_max_tokens: int = 1000
    default_k: int = 24
    default_context_levels: List[str] = DEFAULT_CONTEXT_LEVELS


settings = Settings()
