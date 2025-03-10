import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and Mediapipe FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Blink detection parameters
blink_threshold = 0.25  # EAR threshold for a blink
blink_cooldown = 1.0    # Minimum time between scroll actions (in seconds)
blink_time_window = 0.5  # Time window to detect double blinks (in seconds)
last_blink_time = 0
last_action_time = 0   
blink_count = 0  # Count blinks within the time window

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR) for given eye landmarks."""
    # Vertical landmarks
    top = landmarks[eye_indices[0]].y
    bottom = landmarks[eye_indices[1]].y
    # Horizontal landmarks
    left = landmarks[eye_indices[2]].x
    right = landmarks[eye_indices[3]].x
    # EAR formula
    ear = (abs(top - bottom)) / (abs(left - right))
    return ear

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, [159, 145, 33, 133])  # Left eye landmarks
        right_ear = calculate_ear(landmarks, [386, 374, 362, 263])  # Right eye landmarks
        avg_ear = (left_ear + right_ear) / 2

        # Blink detection logic
        current_time = time.time()         
        if avg_ear < blink_threshold:  # Eyes closed
            if current_time - last_blink_time > 0.2:  # Prevent rapid false positives
                blink_count += 1
                last_blink_time = current_time
                print(f"Blink count: {blink_count}")

        # Handle actions based on blink count
        if blink_count > 0 and current_time - last_blink_time > blink_time_window:
            if blink_count == 1:  # Single blink
                pyautogui.press('space')  # Scroll down
                print("Scrolled Down")
            elif blink_count == 2:  # Double blink
                pyautogui.hotkey('shift', 'space')  # Scroll up
                print("Scrolled Up")

            # Reset after action
            last_action_time = current_time
            blink_count = 0

        # Draw eye landmarks for debugging
        for eye_landmark in [33, 133, 362, 263, 159, 145, 386, 374]:
            x = int(landmarks[eye_landmark].x * frame_w)
            y = int(landmarks[eye_landmark].y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Eye-Controlled Scrolling", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()