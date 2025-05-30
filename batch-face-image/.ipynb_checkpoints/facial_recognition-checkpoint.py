import cv2
import numpy as np
import logging
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
import os

logging.basicConfig(level=logging.INFO)

def preprocess_image(image_path):
    """Preprocess image: Read, convert BGR to RGB."""
    # Use os.path.basename to avoid path traversal issues
    safe_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path))
    image = cv2.imread(safe_path)
    if image is None:
        raise ValueError(f"Unable to read image at path: {safe_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_embeddings(image_path):
    """Extract facial embeddings for a single image."""
    image = preprocess_image(image_path)
    logging.info(f"Processing image: {image_path}")

    try:
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="retinaface",
            enforce_detection=True
        )
        logging.info(f"DeepFace output: {embeddings}")  # Debug log

        if not embeddings:
            logging.warning(f"No face detected in {image_path}.")
            return []

        face_embeddings = []
        for idx, face_data in enumerate(embeddings):
            # Check if 'facial_area' key exists (used in newer versions of DeepFace)
            if "facial_area" in face_data:
                face = face_data["facial_area"]
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            # Fallback to 'region' key (used in older versions of DeepFace)
            elif "region" in face_data:
                face = face_data["region"]
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            else:
                logging.warning(f"No face region data found for face {idx} in {image_path}.")
                continue

            face_embeddings.append({
                "embedding": face_data["embedding"],
                "bbox": [x, y, w, h],
                "face_index": idx  # Add face index to uniquely identify each face
            })
        return face_embeddings
    except Exception as e:
        logging.error(f"Error extracting embeddings: {str(e)}")
        return []


def extract_embeddings_batch(image_paths):
    """Extract embeddings from multiple images in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(extract_embeddings, image_paths))