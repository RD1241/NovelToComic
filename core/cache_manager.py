import hashlib
import sqlite3
import os
import json
from config import settings

class CacheManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(settings.BASE_DIR, "core", "cache.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                hash_key TEXT PRIMARY KEY,
                result_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_cached_result(self, text: str):
        if not settings.ENABLE_CACHING:
            return None
            
        h = self.get_hash(text)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT result_json FROM cache WHERE hash_key = ?', (h,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None

    def set_cached_result(self, text: str, result: dict):
        if not settings.ENABLE_CACHING:
            return
            
        h = self.get_hash(text)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO cache (hash_key, result_json)
            VALUES (?, ?)
        ''', (h, json.dumps(result)))
        conn.commit()
        conn.close()
