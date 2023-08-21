''' 
    Impor Pustaka apa yang kita butuhkan
'''
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time


''' 
    Config Mediapipe Solution
'''
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def diffTime(looking_down_timer):
    diff = (datetime.now() - looking_down_timer).total_seconds()
    return diff 


def putText(frame, val, x, y, unity=""):
    if len(unity) == 0:
        cv2.putText(frame, f'{val}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f'{val:.4f} {unity}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
def getOurLandmark():
    our_landmarks = []
    for idxs in (chosen_left_eye_idxs, chosen_right_eye_idxs, chosen_mouth_idxs, nose_idxs, forehead_idxs):
        our_landmarks.extend(idxs)
    return our_landmarks
        
def getTargettedLandmark():
    targetted_landmark = []
    targetted_landmark.extend(item for sublist in [nose_tip, right_eye, left_eye, lips_upper, chin] for item in sublist)  
    return targetted_landmark    

def headPose(image, face_landmarks, face_2d, face_3d, nose_2d, nose_3d, looking_down_timer, required_looking_time):
    # Mendapatkan dimensi gambar
    img_h, img_w, img_c = image.shape
    targetted_landmark = getTargettedLandmark()

    # Konversi koordinat menjadi array NumPy
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Matriks kamera
    focal_length = 1 * img_w
    cam_matrix = np.array([
                            [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]
                        ])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Hitung orientasi kepala menggunakan solvePnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Matriks Rotasi
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Angle
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # Mendapatkan Derajat Rotasi Y
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    # Melihat kemiringan kepala pengguna
    if y < -10:
        text = 'Looking Left'
    elif y > 10:
        text = 'Looking Right'
    elif x < -7:
        text = 'Looking Down'
    elif x > 10:
        text = 'Looking Up'
    else:
        text = 'Forward'
    
    if text == 'Looking Down' and looking_down_timer == None:
        looking_down_timer = datetime.now()
    elif text != 'Looking Down': 
        looking_down_timer = None
        
    if looking_down_timer is not None :
        diff_time = diffTime(looking_down_timer)
        putText(image, diff_time, 20, 100, 'seconds')
    
        if diff_time > required_looking_time:
            putText(image, "GO SLEEP NOW", 20, 150)
        print(looking_down_timer, '\t', diff_time)
        
    # Memproyeksikan koordinat hidung ke 3d
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    
    # Gambar Arah Pandangan
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    cv2.line(image, p1, p2, (255, 0, 0), 3)
    
    return p1, p2, text, x, y, z 
    
    
    
def praHeadPose(img, idx, lm, nose_2d, nose_3d):
    img_h, img_w, img_c = image.shape
    if idx == 1:
    # Dapatkan koordinat hidung dalam 2D dan 3D
        nose_2d = (lm.x * img_w, lm.y * img_h)
        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
    
    x, y = int(lm.x * img_w), int(lm.y * img_h)
    
    # Dapatkan koordinat 2D
    face_2d.append([x, y])
    # Dapatkan koordinat 3D
    face_3d.append([x, y, lm.z])
    
    return nose_2d, nose_3d

