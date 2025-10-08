import sqlite3
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class Database:
    """
    Simple SQLite database for storing historical Metrics
    In production, replace with PostgreSQL or TimescaleDB
    """

    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table if they dont exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            deployment TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            cpu_usage REAL NOT NULL,
            memory_usage REAL NOT NULL,
            replica_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(namespace, deployment, timestamp) 
            )
            """
            )

            # Create index for faster queries
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_namespace_deployment_timestamp
            ON metrics (namespace, deployment, timestamp DESC) 
            """
            )

            conn.commit()
            logger.info(f"Database iniitalized at {self.db_path}")

    def insert_metric(
        self,
        namespace: str,
        deployment: str,
        timestamp: datetime,
        cpu_usage: float,
        memory_usage: float,
        replica_count: Optional[int] = None,
    ):
        """Insert a new metric record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            INSERT OR REPLACE INTO metrics
            (namespace, deployment, timestamp, cpu_usage, memory_usage, replica_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    namespace,
                    deployment,
                    timestamp,
                    cpu_usage,
                    memory_usage,
                    replica_count,
                ),
            )
            conn.commit()

    def get_metrics(
        self,
        namespace: str,
        deployment: str,
        limit: int = 168,  # Default: last week (hourly data)
    ) -> List[Dict]:
        """
        Retrieve historical metrics for a deployment

        Args:
            namespace: K8s namespace
            deployment: Deployment name
            limit: Number of most recent records to fetch

        Returns:
            List of metric dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT 
                    timestamp,
                    cpu_usage,
                    memory_usage,
                    replica_count
                FROM metrics 
                WHERE namespace = ? AND deployment = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (namespace, deployment, limit),
            )

            rows = cursor.fetchall()

            # Convert to list of dicts and reverse to chronological order
            metrics = [dict(row) for row in rows]
            metrics.reverse()

            return metrics

    from typing import Optional

    def get_latest_metric(self, namespace: str, deployment: str) -> Optional[Dict]:
        """Get the most recent metric for a deployment"""
        metrics = self.get_metrics(namespace, deployment, limit=1)
        return metrics[0] if metrics else None

    def cleanup_old_data(self, days: int = 30):
        """
        Remove metrics older than specified days
        Helps manage database size
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM metrics
                WHERE timestamp < datetime('now', ? || ' days')
                """,
                (f"-{days}",),
            )

            deleted = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted} old records")
            return deleted

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT namespace) as namespaces,
                    COUNT(DISTINCT deployment) as deployments,
                    MIN(timestamp) as oldest_record,
                    MAX(timestamp) as newest_record
                FROM metrics
            """
            )

            row = cursor.fetchone()

            return {
                "total_records": row[0],
                "namespaces": row[1],
                "deployments": row[2],
                "oldest_record": row[3],
                "newest_record": row[4],
            }
