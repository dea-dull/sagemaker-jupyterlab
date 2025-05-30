import numpy as np
from pinecone import Pinecone
import hashlib
import os
from config import API_KEY, INDEX_NAME, ORGANIZATION_NAME
import logging

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

def generate_face_id(image_path, face_index):
    """Generate a unique Face ID using SHA256 hash and face index."""
    try:
        # Debug: Log the image path being processed
        logging.info(f"Generating Face ID for image: {image_path}")

        # Check if the image file exists
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return None

        with open(image_path, "rb") as img_file:
            return f"face_{hashlib.sha256(img_file.read()).hexdigest()[:16]}_{face_index}"
    except Exception as e:
        logging.error(f"Error generating Face ID: {e}")
        return None

def normalize_embedding(embedding):
    """Normalize an embedding to unit length."""
    return embedding / np.linalg.norm(embedding)

def upload_embeddings_to_pinecone(embeddings_list, metadata_list, namespace=ORGANIZATION_NAME):
    """Upload batch of embeddings to Pinecone."""
    vectors = []
    
    for embeddings, metadata in zip(embeddings_list, metadata_list):
        if not embeddings:  # Skip empty embeddings
            logging.warning(f"No embeddings found for {metadata['image_id']}. Skipping upload.")
            continue

        # Construct the full image path
        image_path = os.path.join("/tmp", metadata["image_id"])  # Assuming images are in /tmp

        # Debug: Log the metadata and image path
        logging.info(f"Processing metadata: {metadata}")
        logging.info(f"Full image path: {image_path}")

        for face in embeddings:
            face_id = generate_face_id(image_path, face["face_index"])
            if face_id:
                vectors.append({
                    "id": face_id,
                    "values": normalize_embedding(face["embedding"]).tolist(),
                    "metadata": metadata
                })

    if not vectors:
        logging.error("No valid vectors to upload. Skipping Pinecone upload.")
        return  # Exit function early if no valid data

    try:
        index.upsert(vectors=vectors, namespace=namespace)
        logging.info(f"Uploaded {len(vectors)} face embeddings to Pinecone (Namespace: {namespace})")
    except Exception as e:
        logging.error(f"Error uploading to Pinecone: {e}")