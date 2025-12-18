from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from app.utils.config import settings
from app.services.embeddings import EMBEDDING_DIM
from app.utils.logger import get_logger
import uuid

logger = get_logger(__name__)

COLLECTION_NAME = "documents"

# Initialize Qdrant client
client = QdrantClient(
    url=settings.QDRANT_URL, 
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
)



def initialize_collection():
    """Create or fix collection with correct vector size"""
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        current_size = collection_info.config.params.vectors.size
        
        if current_size != EMBEDDING_DIM:
            logger.warning(f"Wrong vector size detected ({current_size} ≠ {EMBEDDING_DIM}). Recreating collection...")
            client.delete_collection(COLLECTION_NAME)
            raise ValueError("Dimension mismatch")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' ready (size: {current_size})")
            
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}' with vector size {EMBEDDING_DIM}")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        logger.info(f"Collection created with size {EMBEDDING_DIM}")


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
    Search similar vectors — works with qdrant-client v1.10+ (2025)
    """
    try:
        from qdrant_client.models import Filter

        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )

        results = []
        for point in search_result.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}

            results.append({
                "id": point.id,
                "score": point.score,
                "text": text,
                "meta": meta
            })

        logger.info(f"Qdrant search returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Qdrant search failed: {e}", exc_info=True)
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
        # we search and then delete by IDs
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