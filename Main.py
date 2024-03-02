import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the hand gesture map for finger counting
gesture_map = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six"
}

def count_fingers(hand_landmarks):
    # Define landmark indices for thumb, index, middle, ring, and pinky fingers
    finger_tip_indices = [4, 8, 12, 16, 20]
    
    # Initialize finger count
    finger_count = 0
    
    # Check each finger
    for tip_idx in finger_tip_indices:
        # Get the y-coordinate of the finger tip
        tip_y = hand_landmarks.landmark[tip_idx].y
        
        # Get the y-coordinate of the middle knuckle
        knuckle_y = hand_landmarks.landmark[tip_idx - 2].y
        
        # If the finger tip is above the middle knuckle, the finger is extended
        if tip_y < knuckle_y:
            finger_count += 1
    
    return finger_count

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                

                
                # Count fingers and determine hand gesture
                finger_count = count_fingers(hand_landmarks)
                hand_gesture = gesture_map.get(finger_count, "unknown")

                hand_bbox = cv2.boundingRect(
                    np.array([[(lm.x * image.shape[1], lm.y * image.shape[0]) 
                               for lm in hand_landmarks.landmark]]).astype(int))

                text_position = (hand_bbox[0] + hand_bbox[2] // 2 - 50, hand_bbox[1] - 20)
                
                # Display the hand gesture
                cv2.putText(image, hand_gesture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Detection', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
