from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # OpenAI
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key for embeddings and LLM")
    # Groq (free & super fast LLM)
    GROQ_API_KEY: str = Field(..., description="Groq API key — get free at https://console.groq.com/keys")
    
    # Qdrant Vector Database
    QDRANT_URL: str = Field(..., description="Qdrant server URL")
    QDRANT_API_KEY: Optional[str] = Field(None, description="Qdrant API key (optional for local)")
    
    # Redis for conversation memory
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # MongoDB for metadata and bookings
    MONGO_URL: str = Field(..., description="MongoDB connection URL")
    MONGO_DB_NAME: str = Field(default="rag_system", description="MongoDB database name")
    
    # Application settings
    MAX_UPLOAD_SIZE_MB: int = Field(default=10, description="Maximum file upload size in MB")
    DEFAULT_CHUNK_STRATEGY: str = Field(default="fixed", description="Default chunking strategy")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

# Singleton settings instance
settings = Settings()