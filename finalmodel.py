import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.5,
                       max_num_hands=1,
                       min_tracking_confidence=0.5)

vid = cv2.VideoCapture(0)

label = {i: chr(65 + i) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []
    try:
        res, frame = vid.read()
        if not res:
            print('unable to capture the frame')
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
        
            predicted_label = prediction[0]  # Directly use the predicted label
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, predicted_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print('some error occurred:', e)
        continue

vid.release()
cv2.destroyAllWindows()