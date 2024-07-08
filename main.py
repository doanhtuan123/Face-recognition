import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import joblib
import matplotlib.pyplot as plt

# Load the trained SVM model and label encoder
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Define the get_embedding function (you should implement this using FaceNet)
def get_embedding(face_img):
    # Placeholder for face embedding (replace with actual logic)
    return np.random.rand(512)


# Create a function for face recognition
def recognize_face(image_path):
    # Load and preprocess the new image
    img = cv2.imread(image_path)

    if img is None:
        return "Image not found or could not be loaded."

    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the new image using MTCNN
    detector = MTCNN()
    results = detector.detect_faces(img)

    if results:
        # Get the first detected face
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)

        # Extract the face region
        face = img[y:y + h, x:x + w]

        # Resize the face to the target size (160x160)
        face_arr = cv2.resize(face, (160, 160))
        plt.imshow(face_arr)
        plt.show()
        # Get the embedding for the new face
        face_embedding = get_embedding(face_arr)

        # Use the trained SVM model to predict the identity
        predicted_label = svm_model.predict([face_embedding])

        # Decode the predicted label using the label encoder
        predicted_identity = label_encoder.inverse_transform(predicted_label)

        return predicted_identity[0]
    else:
        return "No face detected"


# Example usage:

new_image_path = "test2.jpg"  # Provide the correct path to your image
predicted_identity = recognize_face(new_image_path)

print("Predicted identity:", predicted_identity)
