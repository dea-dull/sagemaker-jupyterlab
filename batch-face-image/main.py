import os
from s3_utils import batch_download_images
from facial_recognition import extract_embeddings_batch
from pinecone_utils import upload_embeddings_to_pinecone
from config import BUCKET_NAME, ORGANIZATION_NAME
from utils import batch_cleanup

BATCH_SIZE = 10  # Process images in batches

def process_batch(s3_keys):
    """Process multiple images in a batch."""
    image_paths = batch_download_images(s3_keys, BUCKET_NAME)
    if not image_paths:
        print("No images downloaded. Exiting.")
        return

    # Process embeddings in parallel
    embeddings_list = extract_embeddings_batch(image_paths)

    # Upload embeddings
    metadata_list = [{"image_id": s3_key, "organization": ORGANIZATION_NAME} for s3_key in s3_keys]
    upload_embeddings_to_pinecone(embeddings_list, metadata_list)

    # Cleanup
    batch_cleanup(image_paths)

def main():
    s3_keys = os.getenv("S3_KEYS", "").split(",")
    for i in range(0, len(s3_keys), BATCH_SIZE):
        process_batch(s3_keys[i:i + BATCH_SIZE])

if __name__ == "__main__":
    main()