from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MongoDBClient:
    """Singleton MongoDB client wrapper"""
    _client: AsyncIOMotorClient = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Get or create MongoDB client"""
        if cls._client is None:
            logger.info(f"Connecting to MongoDB at {settings.MONGO_URL}")
            cls._client = AsyncIOMotorClient(settings.MONGO_URL)
            logger.info("MongoDB client initialized")
        return cls._client

    @classmethod
    def get_database(cls):
        """Get database instance"""
        client = cls.get_client()
        return client[settings.MONGO_DB_NAME]
    
    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls._client:
            cls._client.close()
            logger.info("MongoDB connection closed")

# Database and collections
db = MongoDBClient.get_database()
files_metadata_collection = db["files_metadata"]
bookings_collection = db["bookings"]

async def save_metadata(metadata: dict) -> str:
    """
    Save file metadata to MongoDB.
    
    Args:
        metadata: Dictionary containing file metadata
        
    Returns:
        Inserted document ID as string
    """
    try:
        result = await files_metadata_collection.insert_one(metadata)
        logger.info(f"Saved metadata with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        raise

async def get_metadata_by_filename(filename: str) -> dict:
    """
    Retrieve metadata for a specific file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Metadata document or None if not found
    """
    try:
        result = await files_metadata_collection.find_one({"file_name": filename})
        return result
    except Exception as e:
        logger.error(f"Error retrieving metadata: {str(e)}")
        raise

async def list_all_files(limit: int = 50) -> list:
    """
    List all uploaded files metadata.
    
    Args:
        limit: Maximum number of results
        
    Returns:
        List of metadata documents
    """
    try:
        cursor = files_metadata_collection.find().limit(limit).sort("_id", -1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            results.append(doc)
        return results
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise