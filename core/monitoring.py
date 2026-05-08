import sqlite3
import os
import json
from datetime import datetime
from PIL import Image, ImageStat
from config import settings

class DriftMonitor:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(settings.BASE_DIR, "core", "metrics.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_metrics (
                job_id TEXT PRIMARY KEY,
                timestamp TEXT,
                input_length INTEGER,
                vocab_diversity REAL,
                scene_count INTEGER,
                avg_char_drift_score REAL,
                generation_time_sec REAL,
                success BOOLEAN,
                anomaly_flags TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def analyze_data_drift(self, text: str, scenes: list) -> dict:
        """Analyzes text properties to detect input data drift."""
        words = text.split()
        input_length = len(words)
        vocab_diversity = len(set(words)) / max(1, input_length)
        scene_count = len(scenes)
        return {
            "input_length": input_length,
            "vocab_diversity": vocab_diversity,
            "scene_count": scene_count
        }

    def analyze_character_drift(self, scenes: list) -> float:
        """Calculates consistency score. 1.0 means perfect consistency. 0.0 means total drift."""
        char_descriptions = {}
        drift_scores = []
        
        for scene in scenes:
            for char in scene.get("characters", []):
                name = char.get("name", "").lower()
                desc = char.get("description", "")
                if not name or not desc:
                    continue
                
                if name in char_descriptions:
                    # Calculate Jaccard similarity of words
                    old_words = set(char_descriptions[name].lower().split())
                    new_words = set(desc.lower().split())
                    intersection = old_words.intersection(new_words)
                    union = old_words.union(new_words)
                    score = len(intersection) / max(1, len(union))
                    drift_scores.append(score)
                
                char_descriptions[name] = desc
                
        if not drift_scores:
            return 1.0
        return sum(drift_scores) / len(drift_scores)

    def analyze_image_heuristics(self, image_path: str) -> list:
        """Returns a list of anomaly flags if the image is blank, too dark, or too light."""
        flags = []
        try:
            with Image.open(image_path) as img:
                stat = ImageStat.Stat(img)
                # Check for completely blank/solid images (low variance)
                if sum(stat.var) < 10:
                    flags.append("BLANK_IMAGE")
                # Check for completely dark images
                if sum(stat.mean) < 15:
                    flags.append("DARK_IMAGE")
                # Check for completely blown out bright images
                if sum(stat.mean) > 750: # RGB max is 3*255=765
                    flags.append("BRIGHT_IMAGE")
        except Exception:
            flags.append("CORRUPT_IMAGE")
            
        return flags

    def log_job_metrics(self, job_id: str, text: str, scenes: list, gen_time: float, success: bool, image_paths: list):
        data_metrics = self.analyze_data_drift(text, scenes)
        char_drift = self.analyze_character_drift(scenes)
        
        anomaly_flags = []
        for img_path in image_paths:
            flags = self.analyze_image_heuristics(img_path)
            anomaly_flags.extend(flags)
            
        # Deduplicate flags
        anomaly_flags = list(set(anomaly_flags))
        
        # Simple thresholding logic for logging alerts
        if char_drift < 0.3:
            anomaly_flags.append("SEVERE_CHAR_DRIFT")
        if data_metrics["scene_count"] == 0:
            anomaly_flags.append("NO_SCENES_EXTRACTED")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO job_metrics 
            (job_id, timestamp, input_length, vocab_diversity, scene_count, avg_char_drift_score, generation_time_sec, success, anomaly_flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id, datetime.now().isoformat(), 
            data_metrics["input_length"], data_metrics["vocab_diversity"], data_metrics["scene_count"],
            char_drift, gen_time, success, json.dumps(anomaly_flags)
        ))
        conn.commit()
        conn.close()

    def get_system_health(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get last 50 jobs
        cursor.execute('SELECT success, generation_time_sec, anomaly_flags FROM job_metrics ORDER BY timestamp DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"status": "healthy", "total_jobs_evaluated": 0}
            
        total = len(rows)
        successes = sum(1 for r in rows if r[0])
        failure_rate = (total - successes) / total
        
        avg_time = sum(r[1] for r in rows if r[0]) / max(1, successes)
        
        anomalies = [json.loads(r[2]) for r in rows if r[2]]
        flattened_anomalies = [item for sublist in anomalies for item in sublist]
        
        health_status = "healthy"
        if failure_rate > 0.3:
            health_status = "degraded (high failure rate)"
        elif "BLANK_IMAGE" in flattened_anomalies:
            health_status = "warning (model producing blank images)"
            
        return {
            "status": health_status,
            "total_jobs_evaluated": total,
            "failure_rate": round(failure_rate, 2),
            "avg_generation_time_sec": round(avg_time, 2),
            "recent_anomalies": list(set(flattened_anomalies))
        }
