import os
import cv2
import numpy as np

input_path = '/Users/sid/Desktop/Project/Videos/Frames/Z'
output_path = '/Users/sid/Desktop/Project/Videos/ResizedFrames/Z'

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            resized_img = cv2.resize(img, (128, 128))
            normalized = resized_img / 255.0
            normalized_img_to_save = (normalized * 255).astype(np.uint8)
            save_path = os.path.join(output_path, filename)
            cv2.imwrite(save_path, normalized_img_to_save)
            print(f'Resized and saved: {save_path}')
        else:
            print('Failed to read the image: {filename}')

print('All images have been resized')
