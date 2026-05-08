import sqlite3
import os
from config import settings

class MemoryManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or settings.DB_PATH
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Simple schema to track character visual attributes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def clear_memory(self):
        """Wipes the memory for a fresh generation job."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM characters')
        conn.commit()
        conn.close()

    def add_character(self, name: str, description: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        name_lower = name.lower()

        try:
            # IDENTITY LOCK: First description from Panel 1 is the immutable anchor.
            # ON CONFLICT DO NOTHING ensures later panels never overwrite Panel 1's anchor.
            cursor.execute('''
                INSERT INTO characters (name, description)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING
            ''', (name_lower, description))
            conn.commit()
        except Exception as e:
            print(f"Memory update error: {e}")
        finally:
            conn.close()

    def get_character(self, name: str) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT description FROM characters WHERE name = ?', (name.lower(),))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
        
    def process_scene_characters(self, scene_data: dict):
        """Extracts characters from scene JSON data and saves them to memory."""
        for character in scene_data.get('characters', []):
            name = character.get('name')
            desc = character.get('description')
            if name and desc:
                self.add_character(name, desc)
