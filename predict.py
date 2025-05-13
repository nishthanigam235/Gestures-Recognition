import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load trained model
with open('asl_model.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    prediction = ""

    if result.multi_hand_landmarks:
        landmarks_all_hands = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks_all_hands.extend([lm.x, lm.y, lm.z])

        # Fill with zeros if only one hand
        while len(landmarks_all_hands) < 126:
            landmarks_all_hands.extend([0.0, 0.0, 0.0])

        data_np = np.array(landmarks_all_hands).reshape(1, -1)
        prediction = model.predict(data_np)[0]

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show prediction
    cv2.putText(frame, f'Predicted: {prediction}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
