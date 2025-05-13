import cv2
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Finger tip landmark IDs in MediaPipe
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    fingers_up = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Thumb
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # Fingers (index to pinky)
            for i in range(1, 5):
                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Show finger count
            total_fingers = fingers_up.count(1)
            cv2.putText(img, f'Fingers: {total_fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
