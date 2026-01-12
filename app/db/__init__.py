"""
Database Module.

Provides MongoDB connection management and database access.
"""

from .mongodb import (
    MongoDBClient,
    get_mongodb_client,
    get_database,
    get_traces_collection,
    get_feedback_collection,
    get_evaluations_collection,
    get_prompts_collection,
)

__all__ = [
    "MongoDBClient",
    "get_mongodb_client",
    "get_database",
    "get_traces_collection",
    "get_feedback_collection",
    "get_evaluations_collection",
    "get_prompts_collection",
]
