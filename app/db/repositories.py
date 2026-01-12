"""
MongoDB Repositories.

Provides data access layer for traces, feedback, evaluations, and prompts.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from .mongodb import (
    get_traces_collection,
    get_feedback_collection,
    get_evaluations_collection,
    get_prompts_collection,
)
from .models import (
    TraceDocument,
    SpanDocument,
    FeedbackDocument,
    EvaluationDocument,
    PromptDocument,
)

logger = logging.getLogger(__name__)


class TraceRepository:
    """Repository for trace operations."""
    
    def __init__(self):
        self._collection: Optional[Collection] = None
    
    @property
    def collection(self) -> Optional[Collection]:
        """Get the traces collection."""
        if self._collection is None:
            self._collection = get_traces_collection()
        return self._collection
    
    def save(self, trace: TraceDocument) -> bool:
        """
        Save or update a trace.
        
        Args:
            trace: TraceDocument to save
            
        Returns:
            True if successful
        """
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB not available, trace not saved")
            return False
        
        try:
            result = coll.update_one(
                {"trace_id": trace.trace_id},
                {"$set": trace.to_dict()},
                upsert=True,
            )
            logger.info(f"Saved trace: {trace.trace_id}, tenant: '{trace.tenant_id}', spans: {trace.total_spans}, upserted: {result.upserted_id is not None}")
            return True
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
            return False
    
    def get_by_id(self, trace_id: str) -> Optional[TraceDocument]:
        """Get trace by ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"trace_id": trace_id})
        if doc:
            return TraceDocument.from_dict(doc)
        return None
    
    def get_by_conversation_id(self, conversation_id: str) -> Optional[TraceDocument]:
        """Get trace by conversation ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"conversation_id": conversation_id})
        if doc:
            return TraceDocument.from_dict(doc)
        return None
    
    def list_by_tenant(
        self,
        tenant_id: str,
        limit: int = 50,
        skip: int = 0,
    ) -> List[TraceDocument]:
        """List traces for a tenant."""
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB collection not available for trace retrieval")
            return []
        
        # Log the query for debugging
        logger.debug(f"Querying traces for tenant_id='{tenant_id}', limit={limit}")
        
        cursor = coll.find(
            {"tenant_id": tenant_id}
        ).sort("timestamp", -1).skip(skip).limit(limit)
        
        traces = [TraceDocument.from_dict(doc) for doc in cursor]
        logger.debug(f"Found {len(traces)} traces for tenant '{tenant_id}'")
        
        return traces
    
    def list_recent(self, limit: int = 50) -> List[TraceDocument]:
        """List recent traces."""
        coll = self.collection
        if coll is None:
            return []
        
        cursor = coll.find().sort("timestamp", -1).limit(limit)
        return [TraceDocument.from_dict(doc) for doc in cursor]
    
    def count_by_tenant(self, tenant_id: str) -> int:
        """Count traces for a tenant."""
        coll = self.collection
        if coll is None:
            return 0
        
        return coll.count_documents({"tenant_id": tenant_id})
    
    def delete(self, trace_id: str) -> bool:
        """Delete a trace."""
        coll = self.collection
        if coll is None:
            return False
        
        result = coll.delete_one({"trace_id": trace_id})
        return result.deleted_count > 0


class FeedbackRepository:
    """Repository for feedback operations."""
    
    def __init__(self):
        self._collection: Optional[Collection] = None
    
    @property
    def collection(self) -> Optional[Collection]:
        """Get the feedback collection."""
        if self._collection is None:
            self._collection = get_feedback_collection()
        return self._collection
    
    def save(self, feedback: FeedbackDocument) -> bool:
        """
        Save feedback.
        
        Args:
            feedback: FeedbackDocument to save
            
        Returns:
            True if successful
        """
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB not available, feedback not saved")
            return False
        
        try:
            result = coll.update_one(
                {"id": feedback.id},
                {"$set": feedback.to_dict()},
                upsert=True,
            )
            logger.info(f"Saved feedback: {feedback.id}, tenant: {feedback.tenant_id}, rating: {feedback.rating}, upserted: {result.upserted_id is not None}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def get_by_id(self, feedback_id: str) -> Optional[FeedbackDocument]:
        """Get feedback by ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"id": feedback_id})
        if doc:
            return FeedbackDocument.from_dict(doc)
        return None
    
    def get_by_conversation_id(self, conversation_id: str) -> Optional[FeedbackDocument]:
        """Get feedback by conversation ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"conversation_id": conversation_id})
        if doc:
            return FeedbackDocument.from_dict(doc)
        return None
    
    def list_by_tenant(
        self,
        tenant_id: str,
        limit: int = 50,
    ) -> List[FeedbackDocument]:
        """List feedback for a tenant."""
        coll = self.collection
        if coll is None:
            return []
        
        cursor = coll.find(
            {"tenant_id": tenant_id}
        ).sort("timestamp", -1).limit(limit)
        
        return [FeedbackDocument.from_dict(doc) for doc in cursor]
    
    def list_all(self, limit: int = 100) -> List[FeedbackDocument]:
        """List all feedback."""
        coll = self.collection
        if coll is None:
            return []
        
        cursor = coll.find().sort("timestamp", -1).limit(limit)
        return [FeedbackDocument.from_dict(doc) for doc in cursor]
    
    def count_by_tenant(self, tenant_id: str) -> int:
        """Count feedback for a tenant."""
        coll = self.collection
        if coll is None:
            return 0
        
        try:
            return coll.count_documents({"tenant_id": tenant_id})
        except Exception as e:
            logger.error(f"Failed to count feedback: {e}")
            return 0
    
    def get_feedback_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback statistics."""
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB not available for feedback stats")
            return {"total": 0, "positive": 0, "negative": 0, "correct": 0, "incorrect": 0}
        
        # First do a simple count to verify data exists
        try:
            if tenant_id:
                total_count = coll.count_documents({"tenant_id": tenant_id})
            else:
                total_count = coll.count_documents({})
            logger.info(f"Feedback count for tenant '{tenant_id}': {total_count}")
        except Exception as e:
            logger.error(f"Failed to count feedback: {e}")
            total_count = 0
        
        if total_count == 0:
            return {"total": 0, "positive": 0, "negative": 0, "correct": 0, "incorrect": 0}
        
        match_filter = {"tenant_id": tenant_id} if tenant_id else {}
        
        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "positive": {"$sum": {"$cond": [{"$eq": ["$rating", True]}, 1, 0]}},
                    "negative": {"$sum": {"$cond": [{"$eq": ["$rating", False]}, 1, 0]}},
                    "correct": {"$sum": {"$cond": [{"$eq": ["$reason", "correct"]}, 1, 0]}},
                    "incorrect": {"$sum": {"$cond": [{"$eq": ["$reason", "incorrect"]}, 1, 0]}},
                }
            }
        ]
        
        try:
            result = list(coll.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                logger.info(f"Feedback stats for tenant '{tenant_id}': {stats}")
                return stats
            else:
                # No results, return zeros
                logger.debug(f"No feedback found for tenant {tenant_id}")
                return {"total": 0, "positive": 0, "negative": 0, "correct": 0, "incorrect": 0}
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
        
        return {"total": 0, "positive": 0, "negative": 0, "correct": 0, "incorrect": 0}


class EvaluationRepository:
    """Repository for evaluation operations."""
    
    def __init__(self):
        self._collection: Optional[Collection] = None
    
    @property
    def collection(self) -> Optional[Collection]:
        """Get the evaluations collection."""
        if self._collection is None:
            self._collection = get_evaluations_collection()
        return self._collection
    
    def save(self, evaluation: EvaluationDocument) -> bool:
        """
        Save evaluation.
        
        Args:
            evaluation: EvaluationDocument to save
            
        Returns:
            True if successful
        """
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB not available, evaluation not saved")
            return False
        
        try:
            coll.update_one(
                {"id": evaluation.id},
                {"$set": evaluation.to_dict()},
                upsert=True,
            )
            logger.debug(f"Saved evaluation: {evaluation.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
            return False
    
    def get_by_trace_id(self, trace_id: str) -> Optional[EvaluationDocument]:
        """Get evaluation by trace ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"trace_id": trace_id})
        if doc:
            return EvaluationDocument.from_dict(doc)
        return None
    
    def get_by_conversation_id(self, conversation_id: str) -> Optional[EvaluationDocument]:
        """Get evaluation by conversation ID."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"conversation_id": conversation_id})
        if doc:
            return EvaluationDocument.from_dict(doc)
        return None
    
    def list_by_tenant(
        self,
        tenant_id: str,
        limit: int = 50,
    ) -> List[EvaluationDocument]:
        """List evaluations for a tenant."""
        coll = self.collection
        if coll is None:
            return []
        
        cursor = coll.find(
            {"tenant_id": tenant_id}
        ).sort("timestamp", -1).limit(limit)
        
        return [EvaluationDocument.from_dict(doc) for doc in cursor]
    
    def get_evaluation_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get evaluation statistics."""
        coll = self.collection
        if coll is None:
            return {}
        
        match_filter = {"tenant_id": tenant_id} if tenant_id else {}
        
        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "avg_confidence": {"$avg": "$self_confidence"},
                    "avg_hallucination_risk": {"$avg": "$hallucination_risk"},
                    "avg_relevancy": {"$avg": "$answer_relevancy"},
                    "avg_faithfulness": {"$avg": "$faithfulness"},
                    "avg_latency": {"$avg": "$latency_seconds"},
                    "avg_prompt_similarity": {"$avg": "$avg_similarity_to_past_prompts"},
                    "total_tokens_input": {"$sum": "$tokens_input"},
                    "total_tokens_output": {"$sum": "$tokens_output"},
                }
            }
        ]
        
        try:
            result = list(coll.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                # Calculate total tokens
                stats["total_tokens"] = stats.get("total_tokens_input", 0) + stats.get("total_tokens_output", 0)
                # Round float values
                for key in stats:
                    if isinstance(stats[key], float):
                        stats[key] = round(stats[key], 3)
                return stats
        except Exception as e:
            logger.error(f"Failed to get evaluation stats: {e}")
        
        return {}
    
    def get_today_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get today's evaluation statistics."""
        coll = self.collection
        if coll is None:
            return {}
        
        # Get today's date in ISO format (start of day)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        match_filter = {"timestamp": {"$gte": today_start}}
        if tenant_id:
            match_filter["tenant_id"] = tenant_id
        
        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": None,
                    "total_results": {"$sum": 1},
                    "tokens_input": {"$sum": "$tokens_input"},
                    "tokens_output": {"$sum": "$tokens_output"},
                    "avg_latency": {"$avg": "$latency_seconds"},
                }
            }
        ]
        
        try:
            result = list(coll.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                stats["total_tokens"] = stats.get("tokens_input", 0) + stats.get("tokens_output", 0)
                # Round float values
                for key in stats:
                    if isinstance(stats[key], float):
                        stats[key] = round(stats[key], 3)
                return stats
        except Exception as e:
            logger.error(f"Failed to get today's stats: {e}")
        
        return {"total_results": 0, "tokens_input": 0, "tokens_output": 0, "total_tokens": 0, "avg_latency": 0}
    
    def get_historical_chart_data(self, tenant_id: Optional[str] = None, limit: int = 15) -> Dict[str, List[Any]]:
        """
        Get the last N evaluations for line chart visualization.
        Returns arrays of values suitable for charting.
        """
        coll = self.collection
        if coll is None:
            return self._empty_chart_data()
        
        query = {"tenant_id": tenant_id} if tenant_id else {}
        
        try:
            # Get last N evaluations sorted by timestamp descending
            cursor = coll.find(
                query,
                {
                    "timestamp": 1,
                    "answer_relevancy": 1,
                    "faithfulness": 1,
                    "self_confidence": 1,
                    "hallucination_risk": 1,
                    "latency_seconds": 1,
                    "tokens_input": 1,
                    "tokens_output": 1,
                }
            ).sort("timestamp", -1).limit(limit)
            
            docs = list(cursor)
            
            # Reverse to get chronological order (oldest first for charts)
            docs.reverse()
            
            # Extract arrays for each metric
            timestamps = []
            relevancy = []
            faithfulness = []
            confidence = []
            hallucination = []
            latency = []
            tokens_input = []
            tokens_output = []
            
            for doc in docs:
                ts = doc.get("timestamp", "")
                # Extract just time portion for display (HH:MM)
                if ts and len(ts) >= 16:
                    timestamps.append(ts[11:16])  # Extract HH:MM from ISO timestamp
                else:
                    timestamps.append("")
                
                relevancy.append(round(doc.get("answer_relevancy", 0), 3))
                faithfulness.append(round(doc.get("faithfulness", 0), 3))
                confidence.append(round(doc.get("self_confidence", 0), 3))
                hallucination.append(round(doc.get("hallucination_risk", 0), 3))
                latency.append(round(doc.get("latency_seconds", 0), 3))
                tokens_input.append(doc.get("tokens_input", 0))
                tokens_output.append(doc.get("tokens_output", 0))
            
            return {
                "timestamps": timestamps,
                "relevancy": relevancy,
                "faithfulness": faithfulness,
                "confidence": confidence,
                "hallucination": hallucination,
                "latency": latency,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "count": len(docs),
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical chart data: {e}")
            return self._empty_chart_data()
    
    def _empty_chart_data(self) -> Dict[str, List[Any]]:
        """Return empty chart data structure."""
        return {
            "timestamps": [],
            "relevancy": [],
            "faithfulness": [],
            "confidence": [],
            "hallucination": [],
            "latency": [],
            "tokens_input": [],
            "tokens_output": [],
            "count": 0,
        }
    
    def update_with_feedback(
        self,
        conversation_id: str,
        feedback_id: str,
        feedback_rating: bool,
    ) -> bool:
        """Update evaluation with feedback data."""
        coll = self.collection
        if coll is None:
            return False
        
        try:
            result = coll.update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "feedback_id": feedback_id,
                        "feedback_rating": feedback_rating,
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update evaluation with feedback: {e}")
            return False


class PromptRepository:
    """Repository for prompt operations (for similarity computation)."""
    
    def __init__(self):
        self._collection: Optional[Collection] = None
    
    @property
    def collection(self) -> Optional[Collection]:
        """Get the prompts collection."""
        if self._collection is None:
            self._collection = get_prompts_collection()
        return self._collection
    
    def save(self, prompt: PromptDocument) -> bool:
        """
        Save prompt with embedding.
        
        Args:
            prompt: PromptDocument to save
            
        Returns:
            True if successful
        """
        coll = self.collection
        if coll is None:
            logger.warning("MongoDB not available, prompt not saved")
            return False
        
        try:
            coll.update_one(
                {"prompt_hash": prompt.prompt_hash},
                {"$set": prompt.to_dict()},
                upsert=True,
            )
            logger.debug(f"Saved prompt: {prompt.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save prompt: {e}")
            return False
    
    def get_by_hash(self, prompt_hash: str) -> Optional[PromptDocument]:
        """Get prompt by hash."""
        coll = self.collection
        if coll is None:
            return None
        
        doc = coll.find_one({"prompt_hash": prompt_hash})
        if doc:
            return PromptDocument.from_dict(doc)
        return None
    
    def get_recent_prompts(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PromptDocument]:
        """Get recent prompts for similarity computation."""
        coll = self.collection
        if coll is None:
            return []
        
        query = {"tenant_id": tenant_id} if tenant_id else {}
        cursor = coll.find(query).sort("timestamp", -1).limit(limit)
        
        return [PromptDocument.from_dict(doc) for doc in cursor]
    
    def get_prompts_with_embeddings(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PromptDocument]:
        """Get prompts that have embeddings."""
        coll = self.collection
        if coll is None:
            return []
        
        query = {"embedding": {"$exists": True, "$ne": []}}
        if tenant_id:
            query["tenant_id"] = tenant_id
        
        cursor = coll.find(query).sort("timestamp", -1).limit(limit)
        return [PromptDocument.from_dict(doc) for doc in cursor]
    
    def update_quality_score(
        self,
        prompt_hash: str,
        quality_score: float,
        feedback_rating: Optional[bool] = None,
    ) -> bool:
        """Update prompt quality score based on feedback."""
        coll = self.collection
        if coll is None:
            return False
        
        try:
            update = {"response_quality_score": quality_score}
            if feedback_rating is not None:
                update["feedback_rating"] = feedback_rating
            
            result = coll.update_one(
                {"prompt_hash": prompt_hash},
                {"$set": update}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update prompt quality: {e}")
            return False
    
    def count_prompts(self, tenant_id: Optional[str] = None) -> int:
        """Count prompts."""
        coll = self.collection
        if coll is None:
            return 0
        
        query = {"tenant_id": tenant_id} if tenant_id else {}
        return coll.count_documents(query)


# Singleton instances
_trace_repo: Optional[TraceRepository] = None
_feedback_repo: Optional[FeedbackRepository] = None
_evaluation_repo: Optional[EvaluationRepository] = None
_prompt_repo: Optional[PromptRepository] = None


def get_trace_repository() -> TraceRepository:
    """Get trace repository singleton."""
    global _trace_repo
    if _trace_repo is None:
        _trace_repo = TraceRepository()
    return _trace_repo


def get_feedback_repository() -> FeedbackRepository:
    """Get feedback repository singleton."""
    global _feedback_repo
    if _feedback_repo is None:
        _feedback_repo = FeedbackRepository()
    return _feedback_repo


def get_evaluation_repository() -> EvaluationRepository:
    """Get evaluation repository singleton."""
    global _evaluation_repo
    if _evaluation_repo is None:
        _evaluation_repo = EvaluationRepository()
    return _evaluation_repo


def get_prompt_repository() -> PromptRepository:
    """Get prompt repository singleton."""
    global _prompt_repo
    if _prompt_repo is None:
        _prompt_repo = PromptRepository()
    return _prompt_repo
