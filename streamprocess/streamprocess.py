import boto3
import logging
import cv2
import numpy as np
import os
from botocore.exceptions import ClientError
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
KVS_STREAM_NAME = os.getenv("KVS_STREAM_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Check if required environment variables are set
if not PINECONE_API_KEY or not KVS_STREAM_NAME:
    logging.error("Missing required environment variables: PINECONE_API_KEY, KVS_STREAM_NAME.")
    exit(1)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("deepface")  # Ensure this index uses Euclidean distance

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize AWS KVS Client
kinesis_video_client = boto3.client("kinesisvideo", region_name=AWS_REGION)
kinesis_media_client = boto3.client("kinesis-video-media", region_name=AWS_REGION)

# Get the KVS stream endpoint
try:
    response = kinesis_video_client.describe_stream(StreamName=KVS_STREAM_NAME)
    stream_endpoint = response["StreamInfo"]["DataEndpoint"]
    
    # Now create a media client for accessing the video data
    kinesis_media_client = boto3.client("kinesis-video-media", endpoint_url=stream_endpoint)
except ClientError as e:
    logging.error(f"Error getting media from KVS: {str(e)}")
    exit(1)

# Dictionary to store embeddings for each tracked face {track_id: [embedding1, embedding2, ...]}
embeddings_dict = {}

def process_kvs_stream():
    """Process real-time video frames from AWS KVS"""
    try:
        # Get the media stream from KVS (starting from the most recent frame)
        media_response = kinesis_media_client.get_media(
            StreamName=KVS_STREAM_NAME,
            StartSelector={"StartSelectorType": "NOW"}
        )
        payload = media_response["Payload"]
        
        # Stream the data and process the video frames
        while True:
            # Read a chunk of video data
            chunk = payload.read(1024 * 1024)  # Read in chunks (1MB)
            if not chunk:
                break  # No more data
            
            # Convert the chunk into a frame
            nparr = np.frombuffer(chunk, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Process the frame (face detection, tracking, etc.)
                process_frame(frame)
    except ClientError as e:
        logging.error(f"Error during media streaming: {str(e)}")

def process_frame(frame):
    """Detect, track, and recognize faces in a video frame"""
    # Detect faces using DeepFace
    faces = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=False)
    
    detections = []
    for face in faces:
        facial_area = face["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["h"], facial_area["w"]
        bbox = [x, y, x + w, y + h]  # Convert to (x1, y1, x2, y2)
        detections.append((bbox, 1.0))  # (bounding_box, confidence_score)

    # Update tracker
    tracked_faces = tracker.update_tracks(detections, frame=frame)

    for track in tracked_faces:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks

        track_id = str(track.track_id)  # Convert track ID to string
        bbox = track.to_ltrb()  # Bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue  # Skip if no face detected or face is too small

        # Generate face embedding using DeepFace
        embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
        if embedding:
            embedding = embedding[0]["embedding"]

            # Store embedding in the dictionary
            if track_id not in embeddings_dict:
                embeddings_dict[track_id] = [np.array(embedding, dtype=np.float32)]
            else:
                embeddings_dict[track_id].append(np.array(embedding, dtype=np.float32))

    # Average embeddings for each tracked face
    final_embeddings = {track_id: np.mean(embeddings, axis=0) for track_id, embeddings in embeddings_dict.items()}

    # Query Pinecone with averaged embeddings
    for track_id, avg_embedding in final_embeddings.items():
        person_name, distance_score = query_pinecone(avg_embedding, index, threshold=0.8)
        if person_name:
            logging.info(f"Track ID {track_id} is {distance_score:.4f} Euclidean distance from {person_name}")
        else:
            logging.info(f"No similar face found in the database for Track ID {track_id}.")

def query_pinecone(embedding, index, threshold=0.8):
    """Query Pinecone for similar faces using Euclidean distance."""
    # Normalize the embedding
    embedding_normalized = embedding / np.linalg.norm(embedding)
    embedding_list = embedding_normalized.tolist()

    response = index.query(
        namespace="default",
        vector=embedding_list,
        top_k=1,  # Get the most similar face
        include_values=True,
        include_metadata=True
    )

    if response.get("matches"):
        best_match = response["matches"][0]
        if best_match["score"] <= threshold:  # Euclidean distance: lower is better
            return best_match.get("metadata", {}).get("name", "Unknown"), best_match["score"]
    return None, 0

# Run the KVS stream processing
process_kvs_stream()
