import os
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,
                       max_num_hands=2)

def get_landmarks(image_path):
    landmarks = []
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at path {image_path} could not be read.\n")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmark_array = np.array(landmarks)   
            flatten_array = landmark_array.flatten()
            return flatten_array

DATA_DIR = '/Users/palashpunde/Documents/Sing_Language_Detection'
data = []
labels = []

for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR,class_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {class_name}\n")

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        landmarks = get_landmarks(image_path)
        if landmarks is not None:
            data.append(landmarks)
            labels.append(class_name)

data = np.array(data)
labels = np.array(labels)

np.save('data.npy', data)
np.save('labels.npy', labels)

print("Data and labels have been successfully saved.\n")

