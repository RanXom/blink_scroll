import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and Mediapipe FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Blink detection parameters
blink_threshold = 0.2  # Adjust based on eye aspect ratio
blink_cooldown = 1.0   # Minimum time between blinks in seconds
last_blink_time = time.time()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Coordinates for left eye landmarks (adjust indices as needed)
        left_top = landmarks[159]  # Upper eyelid
        left_bottom = landmarks[145]  # Lower eyelid
        
        # Coordinates for right eye landmarks
        right_top = landmarks[386]
        right_bottom = landmarks[374]

        # Calculate distances
        left_eye_distance = abs(left_top.y - left_bottom.y)
        right_eye_distance = abs(right_top.y - right_bottom.y)

        # Average eye distance to detect blinking
        avg_eye_distance = (left_eye_distance + right_eye_distance) / 2

        # Detect blink
        if avg_eye_distance < blink_threshold:
            current_time = time.time()
            if current_time - last_blink_time > blink_cooldown:
                # Scroll down
                pyautogui.press('space')
                print("Scrolled Down")
                last_blink_time = current_time
        elif avg_eye_distance > blink_threshold:
            current_time = time.time()
            if current_time - last_blink_time > blink_cooldown:
                # Scroll up
                pyautogui.hotkey('shift', 'space')
                print("Scrolled Up")
                last_blink_time = current_time

        # Draw eye landmarks for debugging
        for eye_landmark in [159, 145, 386, 374]:
            x = int(landmarks[eye_landmark].x * frame_w)
            y = int(landmarks[eye_landmark].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Eye-Controlled Scrolling", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
