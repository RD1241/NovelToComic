import os
import shutil
from datetime import datetime, timedelta

def clean_old_outputs(days_to_keep=7):
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print(f"Directory {outputs_dir} does not exist.")
        return

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_count = 0

    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if os.path.isdir(item_path):
            try:
                # Assuming folder names are in format YYYYMMDD_HHMMSS
                folder_time = datetime.strptime(item, "%Y%m%d_%H%M%S")
                if folder_time < cutoff_date:
                    shutil.rmtree(item_path)
                    print(f"Deleted old output: {item}")
                    deleted_count += 1
            except ValueError:
                # Skip folders that don't match the timestamp pattern
                pass

    print(f"Cleanup complete. Deleted {deleted_count} folders older than {days_to_keep} days.")

if __name__ == "__main__":
    clean_old_outputs()
