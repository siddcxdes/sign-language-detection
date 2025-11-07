import cv2
import numpy as np
import os

vid = cv2.VideoCapture(0)
output_folder = "/Users/sidxcodes/Developer/sign-language-detection/dataset/Z"
frame_count = 0
max_frame = 800

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not vid.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

while frame_count < max_frame:
    success, frame = vid.read()
    if not success or frame is None:
        print(f"Unable to access the frame at count {frame_count}")
        break

    frame = cv2.resize(frame, None, fx=0.3, fy=0.3) 
    cv2.imshow("Frame", frame)

    resized_img = cv2.resize(frame, (48, 48))
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    filename = f"{output_folder}/Z_{frame_count}.jpg"

    cv2.imwrite(filename, gray)

    print(f"Saved frame {frame_count} -> {filename}")
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exit requested by user.")
        break

vid.release()
cv2.destroyAllWindows()
print(f"Total frames saved: {frame_count}")
