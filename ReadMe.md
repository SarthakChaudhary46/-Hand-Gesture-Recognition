# Real-time Hand Gesture Recognition

This project demonstrates real-time hand gesture recognition using Python, OpenCV, and MediaPipe. It detects hand landmarks from webcam video input, counts the number of fingers extended, and determines the corresponding hand gesture in real-time.

## Features

- **Hand Tracking**: Utilizes the MediaPipe library to track hand landmarks in real-time.
- **Finger Counting**: Counts the number of fingers extended by analyzing the hand landmarks.
- **Gesture Recognition**: Determines hand gestures based on the finger count and maps them to predefined gestures.
- **Webcam Support**: Works with any standard webcam connected to the system.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- Mediapipe
- NumPy

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/your-username/hand-gesture-recognition.git
    ```

2. Install the required Python dependencies:

    ```
    pip install opencv-python mediapipe numpy
    ```

## Usage

1. Run the Python script:

    ```
    python hand_gesture_recognition.py
    ```

2. Point your hand towards the webcam and extend your fingers to see real-time hand gesture recognition.

3. Press 'q' to quit the application.

