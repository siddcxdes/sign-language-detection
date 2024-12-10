import cv2
import os

video_path = '/Users/sid/Desktop/Project/Videos/Z.mp4'
output_folder = '/Users/sid/Desktop/Project/Videos/Frames/Z'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

vid = cv2.VideoCapture(video_path)
frame_number = 0

while True:
    ret, frame = vid.read()
    if not ret:
        break
    filename = output_folder + "/frame_" + str(frame_number) + ".jpg"
    cv2.imwrite(filename, frame)
    frame_number += 1

vid.release()
print(f"Extracted {frame_number} frames to {output_folder}")