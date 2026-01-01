import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from spellchecker import SpellChecker

model = tf.keras.models.load_model("asl_landmark_mlp.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y])

    return np.array(landmarks).reshape(1, -1)


def predict_letter(landmark_vector):
    preds = model.predict(landmark_vector, verbose=0)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    return chr(idx + ord("A")), confidence


class ASLTranslator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL Translator")
        self.resize(900, 700)

        self.last_letter = None
        self.stable_since = None
        self.hold_time = 2.0

        self.cap = None
        self.camera_on = False

        self.history = []
        self.window = 8

        self.current_text = ""
        self.tts = pyttsx3.init()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        self.text_area = QTextEdit()
        self.text_area.setFixedHeight(120)

        self.confidence_label = QLabel("Confidence")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)

        self.hold_label = QLabel("Hold Time")
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, 100)
        self.hold_bar.setValue(100)

        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.clear_btn = QPushButton("Clear Text")
        self.speak_btn = QPushButton("Speak")

        self.spell = SpellChecker()

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addWidget(self.speak_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(self.hold_label)
        layout.addWidget(self.hold_bar)
        layout.addLayout(buttons)
        layout.addWidget(self.text_area)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.clear_btn.clicked.connect(self.clear_text)
        self.speak_btn.clicked.connect(self.speak_text)

    def start_camera(self):
        if not self.camera_on:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            self.timer.start(30)

    def stop_camera(self):
        if self.camera_on:
            self.timer.stop()
            self.cap.release()
            self.camera_on = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        label_w = self.video_label.width()
        label_h = self.video_label.height()
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        qimg = QImage(
            resized.data,
            new_w,
            new_h,
            new_w * ch,
            QImage.Format_RGB888,
        )

        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        landmarks = extract_landmarks(frame)
        if landmarks is None:
            self.confidence_bar.setValue(0)
            if self.current_text and not self.current_text.endswith(" "):
                self.autocorrect_last_word()
            return

        letter, conf = predict_letter(landmarks)
        self.confidence_bar.setValue(int(conf * 100))

        self.handle_letter(letter)

    def handle_letter(self, letter):
        now = time.time()

        if self.last_letter is None:
            self.last_letter = letter
            self.stable_since = now
            self.hold_bar.setValue(100)
            return

        if letter != self.last_letter:
            self.last_letter = letter
            self.stable_since = now
            self.hold_bar.setValue(100)
            return

        if self.stable_since is not None:
            elapsed = now - self.stable_since
            remaining = max(0.0, self.hold_time - elapsed)
            percent = int((remaining / self.hold_time) * 100)
            self.hold_bar.setValue(percent)
            if elapsed >= self.hold_time:
                self.current_text += letter
                self.text_area.setText(self.current_text)

                self.last_letter = None
                self.stable_since = None
                self.hold_bar.setValue(100)

    def smooth(self, letter):
        self.history.append(letter)
        if len(self.history) < self.window:
            return None
        if len(self.history) > self.window:
            self.history.pop(0)
        return max(set(self.history), key=self.history.count)

    def clear_text(self):
        self.current_text = ""
        self.text_area.clear()

    def speak_text(self):
        self.tts.say(self.current_text)
        self.tts.runAndWait()

    def autocorrect_last_word(self):
        words = self.current_text.strip().split()

        if not words:
            return

        last_word = words[-1]

        if len(last_word) <= 2:
            return

        corrected = self.spell.correction(last_word)

        if corrected and corrected != last_word:
            words[-1] = corrected
            self.current_text = " ".join(words) + " "
            self.text_area.setText(self.current_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ASLTranslator()
    gui.show()
    sys.exit(app.exec_())
