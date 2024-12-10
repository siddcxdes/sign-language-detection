import cv2
import mediapipe as mp
import os
import pickle

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=2, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)

input_folder = '/Users/sid/Desktop/Project/Videos/ResizedFrames' 

data = []
labels = []

for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)
    
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = [] 
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x)
                            data_aux.append(y)
                        
                        data.append(data_aux)
                        labels.append(folder_name)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data extraction and saving complete!")