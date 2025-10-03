import asyncpg
import redis.asyncio as aioredis
from typing import Optional, List, Dict, Any
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, postgres_url: str, redis_url: str):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created")

            # Redis connection
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Close database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()

    # PostgreSQL Operations
    async def get_historical_metrics(
        self,
        resource_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Fetch historical metrics from PostgreSQL"""
        query = """
            SELECT timestamp, value, resource_type
            FROM metrics
            WHERE resource_id = $1
            AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp ASC
            LIMIT $4
        """
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, resource_id, start_time, end_time, limit)
            return [dict(row) for row in rows]

    async def save_prediction(
        self,
        resource_id: str,
        resource_type: str,
        predictions: List[Dict[str, Any]],
        model_used: str,
        accuracy_score: Optional[float] = None
    ) -> int:
        """Save prediction results to PostgreSQL"""
        query = """
            INSERT INTO predictions (
                resource_id, resource_type, prediction_data,
                model_used, accuracy_score, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """
        async with self.pg_pool.acquire() as conn:
            prediction_id = await conn.fetchval(
                query,
                resource_id,
                resource_type,
                json.dumps(predictions),
                model_used,
                accuracy_score,
                datetime.utcnow()
            )
            return prediction_id

    async def save_anomaly(
        self,
        resource_id: str,
        anomaly_data: Dict[str, Any],
        severity: str,
        risk_score: float
    ) -> int:
        """Save detected anomaly to PostgreSQL"""
        query = """
            INSERT INTO anomalies (
                resource_id, anomaly_data, severity,
                risk_score, detected_at
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        async with self.pg_pool.acquire() as conn:
            anomaly_id = await conn.fetchval(
                query,
                resource_id,
                json.dumps(anomaly_data),
                severity,
                risk_score,
                datetime.utcnow()
            )
            return anomaly_id

    # Redis Caching Operations
    async def cache_prediction(
        self,
        resource_id: str,
        prediction_data: Dict[str, Any],
        ttl: int = 3600
    ):
        """Cache prediction in Redis"""
        key = f"prediction:{resource_id}"
        await self.redis_client.setex(
            key,
            ttl,
            json.dumps(prediction_data, default=str)
        )

    async def get_cached_prediction(
        self,
        resource_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction from Redis"""
        key = f"prediction:{resource_id}"
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    async def cache_model_state(
        self,
        model_id: str,
        state_data: bytes,
        ttl: int = 86400
    ):
        """Cache trained model state in Redis"""
        key = f"model:{model_id}"
        await self.redis_client.setex(key, ttl, state_data)

    async def get_cached_model_state(
        self,
        model_id: str
    ) -> Optional[bytes]:
        """Retrieve cached model state from Redis"""
        key = f"model:{model_id}"
        return await self.redis_client.get(key)

    async def increment_prediction_counter(self, model_type: str):
        """Increment prediction counter for monitoring"""
        key = f"counter:predictions:{model_type}"
        await self.redis_client.incr(key)

    async def get_prediction_stats(self) -> Dict[str, int]:
        """Get prediction statistics"""
        keys = await self.redis_client.keys("counter:predictions:*")
        stats = {}
        for key in keys:
            model_type = key.split(":")[-1]
            count = await self.redis_client.get(key)
            stats[model_type] = int(count) if count else 0
        return stats


# Global database instance
db = Database(
    postgres_url="",  # Set from environment
    redis_url=""      # Set from environment
)