import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning,
                         message=".*SymbolDatabase.GetPrototype.*")

model = load_model('sign_language_model.h5')
label_encoder = LabelEncoder()
labels = np.load('labels.npy')
label_encoder.fit(labels)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

word_buffer = []
letter_buffer = []
last_time = time.time()
buffer_time = 5.0

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            class_name = label_encoder.inverse_transform([class_id])[0]

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, class_name, (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                         10, (255, 0, 0), 10, cv2.LINE_AA)

            if class_name:
                current_time = time.time()
                if (current_time - last_time > buffer_time):
                    if class_name not in letter_buffer:
                        letter_buffer.append(class_name)
                        last_time = current_time 

                if len(letter_buffer) > 0:
                    word_buffer.append(''.join(letter_buffer))
                    letter_buffer.clear()

                if len(word_buffer) > 0:
                    word_display = ' '.join(word_buffer)
                    cv2.putText(image, word_display, (50, 450), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detection', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

