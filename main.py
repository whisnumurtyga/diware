import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
ngantuk_flag = False
ngantuk_threshold = 0.02  # Ambang batas jarak bibir atas dan bibir bawah
count_gantuk = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_detection = face_detection.process(frame_rgb)

    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            results_mesh = face_mesh.process(frame_rgb)
            if results_mesh.multi_face_landmarks:
                for landmarks in results_mesh.multi_face_landmarks:
                    lip_upper = landmarks.landmark[13]
                    lip_lower = landmarks.landmark[14]

                    distance = ((lip_upper.x - lip_lower.x)**2 + (lip_upper.y - lip_lower.y)**2)**0.5

                    cv2.putText(frame, f'Lip Distance: {distance:.2f}', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if distance >= ngantuk_threshold:
                        ngantuk_flag = True
                    else:
                        ngantuk_flag = False

                    if ngantuk_flag:
                        cv2.putText(frame, "Ngantuk!", (bbox[0], bbox[1] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            mp_drawing.draw_detection(frame, detection)

    cv2.imshow('Face Lip Distance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
