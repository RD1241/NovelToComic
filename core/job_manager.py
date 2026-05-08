import sqlite3
import os
import uuid
import json
from datetime import datetime
from config import settings

class JobManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(settings.BASE_DIR, "core", "jobs.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                progress TEXT,
                created_at TEXT,
                updated_at TEXT,
                result TEXT,
                error TEXT
            )
        ''')
        # Add progress column if upgrading from old schema
        try:
            cursor.execute("ALTER TABLE jobs ADD COLUMN progress TEXT")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO jobs (job_id, status, progress, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_id, 'queued', 'Waiting in queue...', now, now))
        conn.commit()
        conn.close()
        return job_id

    def update_job(self, job_id: str, status: str, result: str = None, error: str = None, progress: str = None):
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE jobs SET status = ?, updated_at = ?, result = ?, error = ?, progress = ?
            WHERE job_id = ?
        ''', (status, now, result, error, progress, job_id))
        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT job_id, status, progress, created_at, updated_at, result, error
            FROM jobs WHERE job_id = ?
        ''', (job_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "job_id": row[0],
            "status": row[1],
            "progress": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "result": json.loads(row[5]) if row[5] else None,
            "error": row[6],
        }
