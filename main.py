import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# 1. Load trained model
# --------------------------
model = load_model("asl_alphabet_cnn.h5")

# Mapping from index to letters
labels = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]) 

# --------------------------
# 2. Initialize MediaPipe Hands
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # default webcam

IMG_SIZE = 64  # must match training

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
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

                # --------------------------
                # Crop hand bounding box
                # --------------------------
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                pad = 20
                x_min, y_min = max(x_min - pad, 0), max(y_min - pad, 0)
                x_max, y_max = min(x_max + pad, w), min(y_max + pad, h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                # --------------------------
                # Preprocess for CNN
                # --------------------------
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                gray = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

                # --------------------------
                # Predict
                # --------------------------
                prediction = model.predict(gray)
                predicted_class = np.argmax(prediction)
                predicted_letter = labels[predicted_class]

                # --------------------------
                # Display prediction
                # --------------------------
                cv2.putText(frame, f"Detected: {predicted_letter}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2, cv2.LINE_AA)

        # Show webcam
        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
