import streamlit as st
import cv2
import pickle
import numpy as np
import mediapipe as mp
from PIL import Image
import time

# Load the trained model
import os

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.p')

try:
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except Exception as e:
    st.error(f"Error: Could not load the model file 'model.p': {str(e)}")
    st.stop()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Create mapping for predictions
labels_dict = {i: chr(65+i) for i in range(26)}  # 0->A, 1->B, etc.

def process_landmarks(hand_landmarks):
    """Extract and process hand landmarks for model input"""
    data = []
    for landmark in hand_landmarks.landmark:
        # Only use x and y coordinates to match the training data
        data.extend([landmark.x, landmark.y])
    return np.array(data)

def detect_hands(frame):
    """Detect hands and draw landmarks"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            # Extract landmarks for prediction
            landmarks = process_landmarks(hand_landmarks)
            prediction = model.predict([landmarks])[0]
            predicted_letter = labels_dict[prediction]
            
            # Draw prediction on frame
            cv2.putText(
                frame_rgb,
                f"Predicted: {predicted_letter}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
    
    return frame_rgb, results.multi_hand_landmarks

def main():
    st.title("✋ Sign Language Detection")
    st.write("Press 'Start' to begin real-time sign language detection")
    
    # Create a button to start/stop the video feed
    if 'running' not in st.session_state:
        st.session_state.running = False

    if st.button('Start' if not st.session_state.running else 'Stop'):
        st.session_state.running = not st.session_state.running

    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    if st.session_state.running:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera. Please make sure camera permissions are granted.")
                st.stop()
                
            # Set the camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame")
                    break

                # Process frame and detect hands
                processed_frame, hand_landmarks = detect_hands(frame)

                # Display the processed frame
                video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)

                # Add a small delay to control frame rate
                time.sleep(0.1)

        finally:
            cap.release()
    else:
        # Display a placeholder image or message when not running
        st.write("Click 'Start' to begin video capture")

if __name__ == '__main__':
    main()

