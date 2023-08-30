#!/usr/bin/env python
# coding: utf-8

# In[8]:


import mediapipe as mp
import numpy as np
import cv2
import math
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time


# In[9]:


chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
nose_tip = [1]
right_eye = [33]
left_eye = [263]
lips_upper = [61, 291]
chin = [199]
chosen_mouth_idxs = [38, 76, 268, 292, 86, 316 ]

face_data = {
    "right_eyes" : [],
    "left_eyes" : [],
    'mouth' : [],
    'head' : []
}


# In[10]:


facemesh =  mp.solutions.face_mesh
face = facemesh.FaceMesh(
    static_image_mode=False, 
    min_tracking_confidence=0.5, 
    min_detection_confidence=0.5
    )
draw = mp.solutions.drawing_utils


# In[11]:


'''
    OTHER
'''
def putText(frame, val, x, y, unity=""):
    if len(unity) == 0:
        cv2.putText(frame, f'{val}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f'{val:.3f} {unity}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
def getTargettedLandmark():
    targetted_landmark = []
    targetted_landmark.extend(item for sublist in [nose_tip, right_eye, left_eye, lips_upper, chin] for item in sublist)  
    return targetted_landmark   


def drawText(frame, val, x, w, y, h):
    cv2.putText(
        frame, text= str(f'p{val}'), org=(int(x * w), int(y * h)),
        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0),
        thickness=1, lineType=cv2.LINE_AA
    )



''' 
    EAR & MAR 
'''
def EARandMAR(right_eye, left_eye, mouth):
    right_eye = (eucledianDistance(right_eye[2], right_eye[5]) + eucledianDistance(right_eye[4], right_eye[3]))/(2*eucledianDistance(right_eye[0], right_eye[1]))
    left_eye = (eucledianDistance(left_eye[4], left_eye[3]) + eucledianDistance(left_eye[5], left_eye[2]))/(2*eucledianDistance(left_eye[1], left_eye[0]))
    mouth = (eucledianDistance(mouth[0], mouth[2]) + eucledianDistance(mouth[3], mouth[5]))/(2*eucledianDistance(mouth[1], mouth[4]))
    return right_eye, left_eye, mouth


def eucledianDistance(point1, point2):
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)


def ARMeans(data):
    return sum(data)/len(data)


def EARAnalsis(data):
    if len(data['right_eyes']) > 0:
        left_eye, right_eye = ARMeans(data['right_eyes']), ARMeans(data['left_eyes'])
        avgEar = (left_eye + right_eye)/2
        if(right_eye < 0.3 and left_eye < 0.3):
            ear_msg = f'gejala mengantuk | mata tertutup > {close_eye_thresh}s | EAR : {str(avgEar)}'
            print(ear_msg)
        # else:
        #     ear_msg = f'belum memiliki gejala mengantuk by mata | EAR : {str(avgEar)}'
        #     print(ear_msg)
        face_data['right_eyes'] = []
        face_data['left_eyes'] = []


def MARAnalisis(data):
    if len(data['mouth']) > 0:
        mouth = ARMeans(data['mouth'])
        if mouth > 0.5:
            mar_msg = f'gejala mengantuk | menguap > {yawn_thresh}s | MAR : {str(mouth)}' 
            print(mar_msg)
        # else:
        #     mar_msg = 'driver tidak memiliki gejala mengantuk melalui pemantauan mulut | Drowsines Value : ' + str(mouth)
        #     print(mar_msg)
        face_data['mouth'] = []




''' 
    HEAD POSE
'''
def praHeadPose(frame_shape, idx, lm):
    h, w = frame_shape
    global face_2d, face_3d, nose_2d, nose_3d
    
    if idx == 1:
    # Dapatkan koordinat hidung dalam 2D dan 3D
        nose_2d = (lm.x * w, lm.y * h)
        nose_3d = (lm.x * w, lm.y * h, lm.z * 3000)
    
    x, y = int(lm.x * w), int(lm.y * h)
    
    # Dapatkan koordinat 2D
    face_2d.append([x, y])
    # Dapatkan koordinat 3D
    face_3d.append([x, y, lm.z])
    

def headPose(image, face_2d, face_3d, nose_2d, nose_3d): 
    # Mendapatkan dimensi gambar
    img_h, img_w, _ = image.shape

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
    _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Matriks Rotasi
    rmat, _ = cv2.Rodrigues(rot_vec)
    # Angle
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # Mendapatkan Derajat Rotasi Y
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    # Melihat kemiringan kepala pengguna
    if x < -7:
        text = 'Looking Down'
    elif x > 10:
        text = 'Looking Up'
    elif y < -10:
        text = 'Looking Left'
    elif y > 10:
        text = 'Looking Right'
    else:
        text = 'Forward'
    
    global face_data
    if abs(x) < 25: #validasi batas atas dan batas bawah (antisipasi outlier)
        face_data['head'].append(x)
    
    # Gambar Arah Pandangan
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    
    cv2.line(frame, p1, p2, (255, 0, 0), 3)
    
    # Tambahkan text pada gambar
    putText(frame, text, 50, 50)
    putText(frame, x, 450, 50, "x")
    putText(frame, y, 450, 100, "y")
    putText(frame, z, 450, 150, "z")


def headAnalisis(data):
    if len(data['head']) > 0 :
        x_head = np.mean(data['head'])
        if(x_head < -7):
            head_msg = 'gejala mengantuk | head down > {}s | x : {:.3f}'.format(head_down_thresh, x_head)
            print(head_msg)
            
        # else:
        #     ear_msg = f'belum memiliki gejala mengantuk by mata | EAR : {str(avgEar)}'
        #     print(ear_msg)
        face_data['head'] = []




'''
    MAIN
'''
def DrowsinessDetection(data, scheduler):
    # Add the function to be called every 1 minute
    scheduler.add_job(MARAnalisis, 'interval', seconds=(yawn_thresh+0.1), args=(data,))
    scheduler.add_job(EARAnalsis, 'interval', seconds=(close_eye_thresh+0.1), args=(data,))
    scheduler.add_job(headAnalisis, 'interval', seconds=(head_down_thresh+0.1), args=(data,))

    # Start the scheduler
    scheduler.start()


# In[12]:


'''
    CONFIG THRESHOLD WAKTU
'''
yawn_thresh = 5
close_eye_thresh = 2
head_down_thresh = 2

head_down_count, close_eye_count, yawn_count = [0, 0, 0]


# In[13]:


cap =  cv2.VideoCapture(0)

scheduler = BackgroundScheduler()
DrowsinessDetection(face_data, scheduler)


while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = face.process(rgb)

    
    if op.multi_face_landmarks:
        face_2d, face_3d = [[], []]
        nose_2d, nose_3d = [None, None]

        
        for pt in op.multi_face_landmarks:
            # draw.draw_landmarks(frame, i)
            temp = [[],[],[]]
            left_eye, right_eye, mouth, head = [1, 1, 1, 1]

            for no,point in enumerate(pt.landmark):

                praHeadPose((img_h, img_w), no, point)

                if no in chosen_left_eye_idxs:
                    # cv2.circle(frame, (int(point.x * img_w), int(point.y * img_h)), 2, (0, color, 0), -1)
                    drawText(frame, left_eye, point.x, img_w, point.y, img_h)
                    temp[0].append(point)
                    left_eye += 1
                if no in chosen_right_eye_idxs:
                    # cv2.circle(frame, (int(point.x * img_w), int(point.y * img_h)), 2, (color, 0, 0), -1)
                    drawText(frame, right_eye, point.x, img_w, point.y, img_h)
                    temp[1].append(point)
                    right_eye += 1
                if no in chosen_mouth_idxs:
                    # cv2.circle(frame, (int(point.x * img_w), int(point.y * img_h)), 2, (0, 0, color), -1)
                    drawText(frame, mouth, point.x, img_w, point.y, img_h)
                    temp[2].append(point)
                    mouth += 1
                if no in nose_tip:
                    drawText(frame, head, point.x, img_w, point.y, img_h)
                    head += 1


            headPose(frame, face_2d, face_3d, nose_2d, nose_3d)


            if len(temp[0]) > 1:
                right_eye, left_eye, mouth = EARandMAR(temp[0], temp[1], temp[2])    
                avg_ear = (right_eye+left_eye)/2 
                
                putText(frame, avg_ear, 50, 100, "ear")
                putText(frame, mouth, 50, 150, "mar")

                face_data['right_eyes'].append(right_eye)
                face_data['left_eyes'].append(left_eye)
                face_data['mouth'].append(mouth)

            
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("windows", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break


scheduler.shutdown()
cap.release()
cv2.destroyAllWindows()


# In[14]:


scheduler.shutdown()
cap.release()
cv2.destroyAllWindows()


# In[ ]:


print(len(face_data['right_eyes']),len(face_data['left_eyes']),len(face_data['mouth']))


# In[ ]:


print(len(face_data['head']))


# In[ ]:


df = pd.DataFrame(face_data)

