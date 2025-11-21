from typing import List
from openai import OpenAI
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize OpenAI client with new SDK
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def embed_texts(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI API.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use (default: text-embedding-3-large)
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
        
    Raises:
        ValueError: If texts list is empty
        Exception: If API call fails
    """
    if not texts:
        logger.warning("Empty texts list provided to embed_texts")
        return []
    
    # Filter out empty strings
    texts = [t.strip() for t in texts if t and t.strip()]
    
    if not texts:
        logger.warning("All texts were empty after filtering")
        return []
    
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts using {model}")
        
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        logger.debug(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def embed_single_text(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """
    Generate embedding for a single text (convenience wrapper).
    
    Args:
        text: Text string to embed
        model: OpenAI embedding model to use
        
    Returns:
        Single embedding vector
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    embeddings = embed_texts([text], model=model)
    return embeddings[0] if embeddings else []
