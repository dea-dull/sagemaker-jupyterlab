from deepface import DeepFace

# Test with a sample image
image_path = "people.png"
embeddings = DeepFace.represent(img_path=image_path, model_name="Facenet", detector_backend="retinaface", enforce_detection=True)
print(embeddings)