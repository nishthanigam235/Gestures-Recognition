# collect_data.py

import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

data_dir = "asl_data"
os.makedirs(data_dir, exist_ok=True)

label = input("Enter the alphabet you want to record (A-Z): ").upper()

cap = cv2.VideoCapture(0)
counter = 0
max_samples = 100

csv_file = open(f"{data_dir}/{label}.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)

while counter < max_samples:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    landmarks_all_hands = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_all_hands.extend([lm.x, lm.y, lm.z])

        # Fill remaining slots with zeros if only one hand is detected
        while len(landmarks_all_hands) < 126:
            landmarks_all_hands.extend([0.0, 0.0, 0.0])

        landmarks_all_hands.append(label)
        csv_writer.writerow(landmarks_all_hands)
        counter += 1

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'{label}: {counter}/{max_samples}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
