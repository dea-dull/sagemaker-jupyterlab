import cv2
import numpy as np
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open the video file
cap = cv2.VideoCapture("classrooma.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
start_frame = int(4 * fps)  # Start at 3 seconds
end_frame = int(8 * fps)  # End at 10 seconds

embeddings_dict = {}  # Dictionary to store embeddings {track_id: embedding_list}

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Skip frames before the start time
    if frame_count < start_frame:
        continue

    # Stop processing after the end time
    if frame_count > end_frame:
        break

    # Process every 5th frame for efficiency
    if frame_count % 5 == 0:
        try:
            # Resize frame for better performance
            frame_resized = cv2.resize(frame, (640, 360))

            # Detect faces
            faces = DeepFace.extract_faces(frame_resized, detector_backend="retinaface", enforce_detection=False)

            detections = []
            for face in faces:
                facial_area = face["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                bbox = [x, y, x + w, y + h]  # Convert to (x1, y1, x2, y2)

                detections.append((bbox, 1.0))  # (bounding_box, confidence_score)

            # Update tracker with detections
            tracked_faces = tracker.update_tracks(detections, frame=frame_resized)

            for track in tracked_faces:
                if not track.is_confirmed():
                    continue  # Ignore unconfirmed tracks

                track_id = track.track_id  # Unique ID assigned by DeepSORT
                bbox = track.to_ltrb()  # Get bounding box (x1, y1, x2, y2)

                # Crop the detected face
                x1, y1, x2, y2 = map(int, bbox)
                face_crop = frame_resized[y1:y2, x1:x2]

                # Ensure the crop is not empty
                if face_crop.size == 0:
                    continue

                # Get embedding for the tracked face
                embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
                if embedding:
                    embedding = embedding[0]["embedding"]
                    if track_id not in embeddings_dict:
                        embeddings_dict[track_id] = [np.array(embedding, dtype=np.float32)]
                    else:
                        embeddings_dict[track_id].append(np.array(embedding, dtype=np.float32))

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

# Average embeddings for each tracked person
final_embeddings = {track_id: np.mean(embeddings, axis=0) for track_id, embeddings in embeddings_dict.items()}

# Convert dictionary to a NumPy array and save
np.save("tracked_face_embeddings.npy", final_embeddings)
print(f"Saved {len(final_embeddings)} unique face embeddings to tracked_face_embeddings.npy")

# Cleanup
cap.release()
cv2.destroyAllWindows()
