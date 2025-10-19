import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

# --------------------------
# 1. Load trained model
# --------------------------
model = load_model("asl_landmark_mlp(NEW).h5")

# --------------------------
# 2. Load label mapping
# --------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Map indices to labels
labels = [class_indices[str(i)] for i in range(len(class_indices))]

# --------------------------
# 3. Initialize MediaPipe Hands
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # default webcam

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                landmarks = np.array(landmarks).reshape(1, -1)

                # Predict
                prediction = model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_letter = labels[predicted_class]

                # Display prediction
                cv2.putText(frame, f"Detected: {predicted_letter}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Show webcam
        cv2.imshow("ASL Landmarks Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
