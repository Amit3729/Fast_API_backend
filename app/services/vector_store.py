from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from app.utils.config import settings
from app.utils.logger import get_logger
import uuid

logger = get_logger(__name__)

COLLECTION_NAME = "documents"

# Initialize Qdrant client
client = QdrantClient(
    url=settings.QDRANT_URL, 
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
)

# Ensure collection exists
def initialize_collection():
    """Create collection if it doesn't exist"""
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' already exists")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # text-embedding-3-large is 3072
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully")

# Initialize on module load
initialize_collection()

def save_vectors(
    chunks: List[str], 
    vectors: List[List[float]], 
    metadata_base: Dict[str, Any]
) -> List[str]:
    """
    Save text chunks and their embeddings to Qdrant.
    
    Args:
        chunks: List of text chunks
        vectors: List of embedding vectors (must match chunks length)
        metadata_base: Base metadata to attach to all chunks
        
    Returns:
        List of generated point IDs
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length")
    
    if not chunks:
        logger.warning("No chunks to save")
        return []
    
    try:
        points: List[PointStruct] = []
        point_ids: List[str] = []
        
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            # Generate UUID for each point
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Combine base metadata with chunk-specific data
            payload = {
                **metadata_base,
                "text": chunk,
                "chunk_index": idx
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                )
            )
        
        # Upsert to Qdrant
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"Successfully saved {len(points)} vectors to Qdrant")
        logger.debug(f"Operation info: {operation_info}")
        
        return point_ids
        
    except Exception as e:
        logger.error(f"Error saving vectors to Qdrant: {str(e)}")
        raise

def search_similar(
    query_vector: List[float], 
    top_k: int = 4,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Qdrant.
    
    Args:
        query_vector: Query embedding vector
        top_k: Number of results to return
        score_threshold: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        List of dicts with id, score, text, and metadata
    """
    try:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        results = []
        for hit in hits:
            result = {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text") if hit.payload else None,
                "meta": {k: v for k, v in (hit.payload or {}).items() if k != "text"}
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} similar vectors")
        return results
        
    except Exception as e:
        logger.error(f"Error searching vectors in Qdrant: {str(e)}")
        raise

def delete_by_source(source: str) -> int:
    """
    Delete all vectors from a specific source.
    
    Args:
        source: Source identifier (filename)
        
    Returns:
        Number of deleted points
    """
    try:
        # Qdrant doesn't support direct filtering delete in all versions
        # So we search and then delete by IDs
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        
        # Search for all points with this source
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source)
                    )
                ]
            ),
            limit=1000
        )
        
        point_ids = [point.id for point in results[0]]
        
        if point_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} points from source '{source}'")
            return len(point_ids)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error deleting vectors by source: {str(e)}")
        raise