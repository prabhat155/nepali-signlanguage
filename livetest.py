import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import mediapipe as mp

# Load model
model = tf.keras.models.load_model("nepali_sign_language_cnn.h5")

IMG_SIZE = (64, 64)

# Class labels
class_names = [
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

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
mp_drawing = mp.solutions.drawing_utils

# For smoothing predictions
prediction_queue = deque(maxlen=15)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_char = "..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = frame.shape
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            x_coords, y_coords = zip(*coords)
            x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
            x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop, resize and normalize ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            img = cv2.resize(roi, IMG_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions[0])
            prediction_queue.append(predicted_class)

            # Get smoothed prediction
            if prediction_queue:
                common_pred = Counter(prediction_queue).most_common(1)[0][0]
                confidence = np.max(predictions[0])
                predicted_char = class_names[common_pred]
                text = f"Prediction: {predicted_char} ({confidence*100:.1f}%)"
            else:
                text = "Predicting..."

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display prediction
            cv2.putText(
                frame,
                text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Show the frame
    cv2.imshow("Nepali Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
