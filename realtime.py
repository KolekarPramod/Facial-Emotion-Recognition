import cv2
from keras.models import load_model
import numpy as np

# Load the model from the .keras file
model = load_model("emotiondetec.keras")

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image to the correct format
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Ensure the image is reshaped as required by the model
    return feature / 255.0  # Normalize the image

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Start webcam feed
while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    # Process each detected face
    try:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Extract the face from the image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a rectangle around the face

            # Resize the face to 48x48 as required by the model
            face_resized = cv2.resize(face, (48, 48))

            # Extract features for the model and predict the emotion
            img = extract_features(face_resized)
            pred = model.predict(img)

            # Get the label with the highest prediction probability
            prediction_label = labels[pred.argmax()]

            # Display the predicted label above the face in the webcam feed
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Show the video feed with emotion predictions
        cv2.imshow("Facial Emotion Recognition", frame)

        # Press 'ESC' to exit
        if cv2.waitKey(27) & 0xFF == 27:
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        pass

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
