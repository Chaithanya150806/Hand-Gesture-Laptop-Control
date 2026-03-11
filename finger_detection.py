import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Finger tip IDs (except thumb)
finger_tips = [8, 12, 16, 20]

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        fingers_up = []

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Draw hand
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand.landmark

            # Index to Pinky
            for tip in finger_tips:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Count fingers
            count = sum(fingers_up)

            cv2.putText(
                frame,
                f"Fingers Up: {count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Day 3 - Finger Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
