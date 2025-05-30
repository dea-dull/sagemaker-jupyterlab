import boto3
import os
import logging
import tempfile
import botocore.exceptions
from concurrent.futures import ThreadPoolExecutor

# Initialize S3 client
s3_client = boto3.client('s3')

def download_image(s3_key, bucket_name):
    """Download a single image from S3."""
    # Use os.path.basename to avoid path traversal
    download_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    try:
        s3_client.download_file(bucket_name, s3_key, download_path)
        logging.info(f"Downloaded: {s3_key}")
        return download_path
    except Exception as e:
        logging.error(f"Error downloading {s3_key}: {e}")
        return None

def batch_download_images(s3_keys, bucket_name):
    """Download multiple images in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(filter(None, executor.map(lambda key: download_image(key, bucket_name), s3_keys)))