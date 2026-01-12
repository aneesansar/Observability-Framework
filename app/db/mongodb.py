"""
MongoDB Connection Management.

Provides thread-safe MongoDB client and collection access.
"""

import logging
from threading import Lock
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    Thread-safe MongoDB client singleton.
    
    Manages connection lifecycle and provides access to collections.
    """
    
    _instance: Optional["MongoDBClient"] = None
    _lock = Lock()
    
    def __new__(cls) -> "MongoDBClient":
        """Ensure singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the MongoDB client."""
        if self._initialized:
            return
        
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._settings = get_settings()
        self._connection_lock = Lock()
        self._initialized = True
        
        logger.info("MongoDBClient initialized")
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._connection_lock:
            if self._client is not None:
                return True
            
            try:
                self._client = MongoClient(
                    self._settings.mongo_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                )
                # Test connection
                self._client.admin.command('ping')
                self._db = self._client[self._settings.mongo_db]
                
                # Create indexes
                self._create_indexes()
                
                logger.info(
                    f"Connected to MongoDB: {self._settings.mongo_uri}, "
                    f"database: {self._settings.mongo_db}"
                )
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"Failed to connect to MongoDB: {e}")
                self._client = None
                self._db = None
                return False
            except Exception as e:
                logger.error(f"Unexpected error connecting to MongoDB: {e}")
                self._client = None
                self._db = None
                return False
    
    def _create_indexes(self) -> None:
        """Create indexes for collections."""
        if self._db is None:
            return
        
        settings = self._settings
        
        try:
            # Traces collection indexes
            traces = self._db[settings.mongo_traces_coll]
            traces.create_index([("trace_id", ASCENDING)], unique=True)
            traces.create_index([("tenant_id", ASCENDING)])
            traces.create_index([("conversation_id", ASCENDING)])
            traces.create_index([("timestamp", DESCENDING)])
            
            # Feedback collection indexes
            feedback = self._db[settings.mongo_feedback_coll]
            feedback.create_index([("id", ASCENDING)], unique=True)
            feedback.create_index([("conversation_id", ASCENDING)])
            feedback.create_index([("tenant_id", ASCENDING)])
            feedback.create_index([("timestamp", DESCENDING)])
            
            # Evaluations collection indexes
            evaluations = self._db[settings.mongo_evaluations_coll]
            evaluations.create_index([("trace_id", ASCENDING)])
            evaluations.create_index([("conversation_id", ASCENDING)])
            evaluations.create_index([("tenant_id", ASCENDING)])
            evaluations.create_index([("timestamp", DESCENDING)])
            
            # Prompts collection indexes (for similarity computation)
            prompts = self._db[settings.mongo_prompts_coll]
            prompts.create_index([("prompt_hash", ASCENDING)], unique=True)
            prompts.create_index([("tenant_id", ASCENDING)])
            prompts.create_index([("timestamp", DESCENDING)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        with self._connection_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._db = None
                logger.info("Disconnected from MongoDB")
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        if self._client is None:
            return False
        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def get_database(self) -> Optional[Database]:
        """
        Get the database instance.
        
        Returns:
            Database instance or None if not connected
        """
        if not self.is_connected():
            self.connect()
        return self._db
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """
        Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance or None if not connected
        """
        db = self.get_database()
        if db is not None:
            return db[name]
        return None
    
    def get_traces_collection(self) -> Optional[Collection]:
        """Get the traces collection."""
        return self.get_collection(self._settings.mongo_traces_coll)
    
    def get_feedback_collection(self) -> Optional[Collection]:
        """Get the feedback collection."""
        return self.get_collection(self._settings.mongo_feedback_coll)
    
    def get_evaluations_collection(self) -> Optional[Collection]:
        """Get the evaluations collection."""
        return self.get_collection(self._settings.mongo_evaluations_coll)
    
    def get_prompts_collection(self) -> Optional[Collection]:
        """Get the prompts collection."""
        return self.get_collection(self._settings.mongo_prompts_coll)


# Global singleton instance
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """
    Get the global MongoDB client instance.
    
    Returns:
        MongoDBClient singleton
    """
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
        _mongodb_client.connect()
    return _mongodb_client


def get_database() -> Optional[Database]:
    """Get the database instance."""
    return get_mongodb_client().get_database()


def get_traces_collection() -> Optional[Collection]:
    """Get the traces collection."""
    return get_mongodb_client().get_traces_collection()


def get_feedback_collection() -> Optional[Collection]:
    """Get the feedback collection."""
    return get_mongodb_client().get_feedback_collection()


def get_evaluations_collection() -> Optional[Collection]:
    """Get the evaluations collection."""
    return get_mongodb_client().get_evaluations_collection()


def get_prompts_collection() -> Optional[Collection]:
    """Get the prompts collection."""
    return get_mongodb_client().get_prompts_collection()
