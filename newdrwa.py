import cv2
import numpy as np
from pinecone import Pinecone
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6m9iGK_SJCAHA6WqusNpxon1s3M9W6VAjfZk86H7mQrrGwQPA67G2nVkhkKLCWcDEas6Kv")
index = pc.Index("deepface")  # Ensure this index uses Euclidean distance

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open the video file
cap = cv2.VideoCapture("classrooma.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(4 * fps)
end_frame = int(8 * fps)

frame_count = 0

# Dictionary to store embeddings for each tracked face {track_id: [embedding1, embedding2, ...]}
embeddings_dict = {}

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
    print(f"Pinecone Response: {response}")  # Debugging: Print the full response

    if response.get("matches"):
        best_match = response["matches"][0]
        print(f"Best Match Score: {best_match['score']}")  # Debugging: Print the score
        if best_match["score"] <= threshold:  # Euclidean distance: lower is better
            return best_match.get("metadata", {}).get("name", "Unknown"), best_match["score"]
    return None, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Skip frames before start time
    if frame_count < start_frame:
        continue
    # Stop processing after the end time
    if frame_count > end_frame:
        break

    # Process every 5th frame for efficiency
    if frame_count % 5 == 0:
        try:
            frame_resized = cv2.resize(frame, (640, 360))

            # Detect faces
            faces = DeepFace.extract_faces(frame_resized, detector_backend="retinaface", enforce_detection=False)

            detections = []
            for face in faces:
                facial_area = face["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["h"], facial_area["w"]
                bbox = [x, y, x + w, y + h]  # Convert to (x1, y1, x2, y2)
                detections.append((bbox, 1.0))  # (bounding_box, confidence_score)

            # Update tracker
            tracked_faces = tracker.update_tracks(detections, frame=frame_resized)

            for track in tracked_faces:
                if not track.is_confirmed():
                    continue  # Ignore unconfirmed tracks

                track_id = str(track.track_id)  # Convert track ID to string
                bbox = track.to_ltrb()  # Bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, bbox)
                face_crop = frame_resized[y1:y2, x1:x2]

                if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue  # Skip if no face detected or face is too small

                # Generate face embedding
                embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
                if embedding:
                    embedding = embedding[0]["embedding"]

                    # Store embedding in the dictionary
                    if track_id not in embeddings_dict:
                        embeddings_dict[track_id] = [np.array(embedding, dtype=np.float32)]
                    else:
                        embeddings_dict[track_id].append(np.array(embedding, dtype=np.float32))

        except Exception as e:
            logging.error(f"Error processing frame {frame_count}: {str(e)}")

# Average embeddings for each tracked face
final_embeddings = {track_id: np.mean(embeddings, axis=0) for track_id, embeddings in embeddings_dict.items()}

# Query Pinecone with averaged embeddings
for track_id, avg_embedding in final_embeddings.items():
    print(f"Averaged Embedding for Track ID {track_id}: {avg_embedding}")  # Debugging: Print the averaged embedding
    person_name, distance_score = query_pinecone(avg_embedding, index, threshold=0.8)
    if person_name:
        logging.info(f"Track ID {track_id} is {distance_score:.4f} Euclidean distance from {person_name}")
    else:
        logging.info(f"No similar face found in the database for Track ID {track_id}.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
