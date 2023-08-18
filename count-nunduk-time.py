import cv2
import mediapipe as mp
import threading
import time

# Initialize MediaPipe FaceMesh and Pose
mp_face_mesh = mp.solutions.face_mesh

# Initialize the FaceMesh module
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize normal chin position (dagu normal)
normal_chin = None
change_detected = False
change_frame_count = 0
looking_down_timer = 0  # Initialize the timer

# Capture video from camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face landmark detection using MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if normal_chin is None:
                # Set normal chin position on first frame
                normal_chin = (face_landmarks.landmark[8].x, face_landmarks.landmark[8].y)

            # Extract coordinates of chin
            chin = (face_landmarks.landmark[8].x, face_landmarks.landmark[8].y)

            # Check for drastic change in chin position
            if abs(chin[1] - normal_chin[1]) > normal_chin[1] * 0.3:
                if change_detected == False:
                    change_frame_count = 60  # Observe for the next 60 frames
                    looking_down_timer = time.time()  # Start the timer when change is detected
                change_detected = True
                print(f'kamera berubah\t{abs(chin[1] - normal_chin[1])}\t{normal_chin[1] * 0.3}')
            else:
                change_frame_count = 60
                change_detected = False
            
            # Perform calibration if change was detected
            if change_detected:
                if change_frame_count > 0:
                    change_frame_count -= 1
                    print(f'-===  change_detected: {change_frame_count}  ===-')
                else:
                    normal_chin = chin
                    change_detected = False

            # Compare current chin position with normal chin position
            range_normal_chin = normal_chin[1] + (normal_chin[1] * 0.15)
            if chin[1] > range_normal_chin:
                status = "Looking down"
            else:
                status = "Not looking down"

            # Calculate the duration of looking down
            if status == "Looking down":
                duration = time.time() - looking_down_timer
                status += f" ({duration:.2f}s)"

            # Draw status on the frame
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(normal_chin[1], "\t", chin[1], '\t', range_normal_chin)

            # Draw landmarks on the frame
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Head Angle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
