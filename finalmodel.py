import cv2
import numpy as np
import pickle
import os
import sys
import importlib.util

# Function to safely import mediapipe
def import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        print("Installing mediapipe...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.8.11"])
        import mediapipe as mp
        return mp

# Import mediapipe safely
mp = import_mediapipe()

# Use os.path for robust file path handling
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.p')
try:
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    raise


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe components with fallback options
def initialize_hands():
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use simpler model for better compatibility
        )
        return hands, mp_hands
    except Exception as e:
        print(f"Error initializing MediaPipe Hands: {e}")
        raise

# Initialize video capture with error handling
def initialize_video():
    try:
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            # Try alternative video source
            vid = cv2.VideoCapture(-1)
            if not vid.isOpened():
                raise ValueError("Could not open any video capture device")
        return vid
    except Exception as e:
        print(f"Error opening video capture: {e}")
        raise

# Initialize components
hands, mp_hands = initialize_hands()
vid = initialize_video()

label = {i: chr(65 + i) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []
    try:
        res, frame = vid.read()
        if not res or frame is None:
            print('Unable to capture the frame')
            break
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            
            prediction = model.predict([np.asarray(data_aux)])
        
            predicted_number = prediction[0]  # Get the predicted number (0-25)
            predicted_label = label[predicted_number]  # Convert to letter using the label dictionary
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, str(predicted_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print('some error occurred:', e)
        continue

# Cleanup
try:
    if 'vid' in locals():
        vid.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error during cleanup: {e}")
finally:
    print("Application terminated")