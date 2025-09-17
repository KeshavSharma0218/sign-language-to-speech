import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# 1. Load trained model
# --------------------------
model = load_model("asl_mnist_cnn.h5")

# Mapping from index to letters (A-Z)
letter_map = {i: chr(65+i) for i in range(26)}  # 0->A, 1->B, ..., 25->Z

# --------------------------
# 2. Initialize MediaPipe Hands
# --------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # 0 = default webcam

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # single hand for simplicity
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
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # --------------------------
                # 3. Crop hand bounding box
                # --------------------------
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = y_max = 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y

                # Add some padding
                pad = 20
                x_min = max(x_min - pad, 0)
                y_min = max(y_min - pad, 0)
                x_max = min(x_max + pad, w)
                y_max = min(y_max + pad, h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                # --------------------------
                # 4. Preprocess for CNN
                # --------------------------
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (28, 28))
                gray = gray.reshape(1, 28, 28, 1).astype("float32") / 255.0

                # --------------------------
                # 5. Predict
                # --------------------------
                prediction = model.predict(gray)
                predicted_class = np.argmax(prediction)
                predicted_letter = letter_map[predicted_class]

                # --------------------------
                # 6. Display prediction
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
