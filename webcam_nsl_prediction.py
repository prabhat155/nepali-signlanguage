import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)

# Define Nepali characters (assuming 36 classes as per the dataset structure)
NEPALI_CHARACTERS = [
    "ka",
    "kha",
    "ga",
    "gha",
    "nga",
    "cha",
    "chha",
    "ja",
    "jha",
    "nya",
    "ta",
    "tha",
    "da",
    "dha",
    "na",
    "ta",
    "tha",
    "da",
    "dha",
    "na",
    "pa",
    "pha",
    "ba",
    "bha",
    "ma",
    "ya",
    "ra",
    "la",
    "wa",
    "sha",
    "shha",
    "sa",
    "ha",
    "ksha",
    "tra",
    "gya",
]

# Load the trained model and scaler
model = tf.keras.models.load_model(
    "nsl_model.h5"
)  # Ensure the model file is in the same directory
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Function to extract hand landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks)
    return None


# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Extract landmarks
        landmarks = extract_landmarks(frame)

        # If landmarks are detected, make prediction
        if landmarks is not None:
            # Scale the landmarks
            landmarks_scaled = scaler.transform([landmarks])

            # Predict the character
            prediction = model.predict(landmarks_scaled)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_char = NEPALI_CHARACTERS[predicted_class]
            confidence = np.max(prediction)

            # Draw landmarks on the frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Display prediction and confidence
            cv2.putText(
                frame,
                f"Character: {predicted_char} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Display the frame
        cv2.imshow("Nepali Sign Language Recognition", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
