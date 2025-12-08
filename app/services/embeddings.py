from typing import List
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

#FREE LOCAL EMBEDDINGS (As openai have limit and take charges on later on)
try:
    from sentence_transformers import SentenceTransformer
    
    
    _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  
    EMBEDDING_DIM = 384
    logger.info("Local embedding model loaded: all-MiniLM-L6-v2 (384-dim, 100% free & private)")

except ImportError as e:
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers torch")
    raise ImportError("Please install: pip install sentence-transformers torch") from e


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using FREE local model (no OpenAI, no internet needed)
    """
    if not texts:
        logger.warning("Empty texts list provided to embed_texts")
        return []

    # Filter empty
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        logger.warning("All texts were empty after filtering")
        return []

    try:
        logger.info(f"Generating LOCAL embeddings for {len(texts)} chunks (all-MiniLM-L6-v2)")
        embeddings = _model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        embeddings_list = embeddings.tolist()
        logger.info(f"Generated {len(embeddings_list)} local embeddings (dim: {EMBEDDING_DIM})")
        return embeddings_list

    except Exception as e:
        logger.error(f"Local embedding failed: {e}", exc_info=True)
        raise


def embed_single_text(text: str) -> List[float]:
    """Convenience for single text"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    return embed_texts([text.strip()])[0]