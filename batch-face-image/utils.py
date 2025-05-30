import os
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

def cleanup_file(file_path):
    """Delete a single file."""
    try:
        # Use os.path.basename to avoid path traversal
        safe_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path))
        os.remove(safe_file_path)
        logging.info(f"Deleted: {safe_file_path}")
    except FileNotFoundError:
        logging.warning(f"File not found: {safe_file_path}")
    except Exception as e:
        logging.error(f"Error deleting {safe_file_path}: {e}")

def batch_cleanup(files):
    """Delete multiple files in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(cleanup_file, files)