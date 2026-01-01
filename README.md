# Sign Language to Text & Speech Translator

A real-time American Sign Language (ASL) translator that converts hand gestures into text and speech using a trained MLP neural network, MediaPipe hand tracking, and a PyQt5 desktop interface.

---

## Features

-  Live webcam ASL recognition
-  MediaPipe hand landmark tracking
-  MLP neural network
-  Real-time text transcription
-  Hold-to-confirm letter system
-  Confidence bar per prediction
-  Word-level autocorrect
-  Text-to-speech output
-  Desktop GUI

## Requirements
- Python 3.9+
- Webcam
- macOS / Windows / Linux //Made on macOS and currently untested on Windows and Linux but it should work on any platform that supports Python 3.9+ and a webcam.

# Setup Instructions

## Clone or Download the project
gh repo clone KeshavSharma0218/sign-language-to-speech
or
git clone https://github.com/KeshavSharma0218/sign-language-to-speech.git
or 
Download and extract the zip 

## (Optional) Create a virtual environment
python3 -m venv venv
macOS / Linux:
source venv/bin/activate
Windows:
venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt
or
pip install tensorflow numpy opencv-python mediapipe PyQt5 pyttsx pyspellchecker

## Run the application
python main.py

Hopefully I will add more features in the future. :)
Todo before 1.0.0
- Fix Issue with the speech button only being called once because of the threading issue
