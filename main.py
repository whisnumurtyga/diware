#!/usr/bin/env python
# coding: utf-8

# In[7]:


import mediapipe as mp
import numpy as np
import cv2
import math
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time


# In[8]:


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
    'mouth' : []
    }


# In[9]:


facemesh =  mp.solutions.face_mesh
face = facemesh.FaceMesh(
    static_image_mode=False, 
    min_tracking_confidence=0.5, 
    min_detection_confidence=0.5
    )
draw = mp.solutions.drawing_utils


# In[10]:


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
    global message
    left_eye, right_eye = ARMeans(data['right_eyes']), ARMeans(data['left_eyes'])
    avgEar = (left_eye + right_eye)/2
    global messageEar
    if(right_eye < 0.3 and left_eye < 0.3):
        message = f'driver mengalami gejala mengantuk karena mata tertutup selama 2 detik | Drowsines Value : {str(left_eye)} {str(right_eye)}'
        print(message)
        messageEar = f'Ngantuk\tMAR: {avgEar}'
    else:
        message = f'driver tidak memiliki gejala mengantuk melalui pemantauan mata | Drowsines Value : {str(left_eye)} {str(right_eye)}'
        print(message)
        messageEar = f'Tidak Ngantuk\tMAR: {avgEar}'
    face_data['right_eyes'] = []
    face_data['left_eyes'] = []
    setEarValue(convertStr(avgEar))


def MARAnalisis(data):
    mouth = ARMeans(data['mouth'])
    setMarValue(convertStr(mouth))
    global messageMar
    if(
        mouth > 0.5
       ):
        message2 = f'driver mengalami gejala mengantuk karena menguap | Drowsines Value : ' + str(mouth)
        messageMar = f'Ngantuk\tMAR: {mouth}'
        print(message2)
    else:
        message2 = 'driver tidak memiliki gejala mengantuk melalui pemantauan mulut | Drowsines Value : ' + str(mouth)
        messageMar = f'Tidak Ngantuk\tMAR: {mouth}'
        print(message2)
    face_data['mouth'] = []

def DrowsinessDetection(data, scheduler):

    # Add the function to be called every 1 minute
    scheduler.add_job(MARAnalisis, 'interval', seconds=10, args=(data,))
    scheduler.add_job(EARAnalsis, 'interval', seconds=2, args=(data,))

    # Start the scheduler
    scheduler.start()


# * Head Movement

def diffTime(looking_down_timer):
    diff = (datetime.now() - looking_down_timer).total_seconds()
    return diff 


def putText(frame, val, x, y, unity=""):
    if len(unity) == 0:
        cv2.putText(frame, f'{val}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f'{val:.2f} {unity}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
def getTargettedLandmark():
    targetted_landmark = []
    targetted_landmark.extend(item for sublist in [nose_tip, right_eye, left_eye, lips_upper, chin] for item in sublist)  
    return targetted_landmark    

def headPose(image, face_2d, face_3d, nose_2d, nose_3d, required_looking_time):
    # Mendapatkan dimensi gambar
    img_h, img_w, _ = image.shape
    # targetted_landmark = getTargettedLandmark()

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
        
    global looking_down_timer
    
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
    
    # Gambar Arah Pandangan
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    cv2.line(image, p1, p2, (255, 0, 0), 3)
    
    return p1, p2, text, x, y, z 
    
    
    
def praHeadPose(img, idx, lm):
    img_h, img_w, _ = img.shape
    global face_2d, face_3d, nose_2d, nose_3d
    if idx == 1:
    # Dapatkan koordinat hidung dalam 2D dan 3D
        nose_2d = (lm.x * img_w, lm.y * img_h)
        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
    
    x, y = int(lm.x * img_w), int(lm.y * img_h)
    
    # Dapatkan koordinat 2D
    face_2d.append([x, y])
    # Dapatkan koordinat 3D
    face_3d.append([x, y, lm.z])
    
    
def setMarValue(val):
    global marVal
    marVal = (val + '  MAR')
    
    
def setEarValue(val):
    global earVal
    earVal = (val + '  EAR')
    
def convertStr(val):
    val = f"{val:.4f}"
    return str(val)


# In[11]:


cap =  cv2.VideoCapture(0)
scheduler = BackgroundScheduler()
DrowsinessDetection(face_data, scheduler)
looking_down_timer = None
earVal = ""
marVal = ""
messageEar = ""
messageMar = ""

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgH, imgW, _ = frame.shape
    # print(imgH, imgW)
    rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = face.process(rgb)
    if op.multi_face_landmarks:

        face_2d = []
        face_3d = []
        nose_2d, nose_3d = [None, None]
        required_looking_time = 2

        for pt in op.multi_face_landmarks:
            # draw.draw_landmarks(frame, i)
            temp = [[],[],[]]
            left = 1
            right = 1
            mouth = 1
            head = 1
            for no,point in enumerate(pt.landmark):

                praHeadPose(frame, no, point)

                if no in chosen_left_eye_idxs:
                    # cv2.circle(frame, (int(point.x * imgW), int(point.y * imgH)), 2, (0, color, 0), -1)
                    cv2.putText(
                        frame, text= str(f'p{left}'), org=(int(point.x * imgW), int(point.y * imgH)),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0),
                        thickness=1, lineType=cv2.LINE_AA
                    )
                    temp[0].append(point)
                    left += 1
                if no in chosen_right_eye_idxs:
                    # cv2.circle(frame, (int(point.x * imgW), int(point.y * imgH)), 2, (color, 0, 0), -1)
                    cv2.putText(
                        frame, text= str(f'p{right}'), org=(int(point.x * imgW), int(point.y * imgH)),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0),
                        thickness=1, lineType=cv2.LINE_AA
                    )
                    temp[1].append(point)
                    right += 1
                if no in chosen_mouth_idxs:
                    # cv2.circle(frame, (int(point.x * imgW), int(point.y * imgH)), 2, (0, 0, color), -1)
                    cv2.putText(
                        frame, text= str(f'p{mouth}'), org=(int(point.x * imgW), int(point.y * imgH)),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0),
                        thickness=1, lineType=cv2.LINE_AA
                    )
                    temp[2].append(point)
                    mouth += 1
                # if no in nose_tip or no in right_eye or no in left_eye or no in lips_upper or no in chin:
                #     cv2.putText(
                #         frame, text= str(f'p{head}'), org=(int(point.x * imgW), int(point.y * imgH)),
                #         fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0),
                #         thickness=1, lineType=cv2.LINE_AA
                #     )
                #     head += 1


            p1, p2, text, x, y, z = headPose(frame, face_2d, face_3d, nose_2d, nose_3d, required_looking_time)
            
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            # Tambahkan text pada gambar
            putText(frame, text, 20, 50)
            putText(frame, x, 450, 50, "x")
            putText(frame, y, 450, 100, "y")
            putText(frame, z, 450, 150, "z")

            putText(frame, earVal, 450, 200)
            putText(frame, marVal, 450, 250)
            print(f'From EAR: {messageEar}')
            print(f'From MAR: {messageMar}')

            right_eye, left_eye, mouth = EARandMAR(temp[0], temp[1], temp[2])
            # print(temp[0], '\t', temp[1], '\t', temp[2])
            face_data['right_eyes'].append(right_eye)
            face_data['left_eyes'].append(left_eye)
            face_data['mouth'].append(mouth)

            
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("windows", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    # break

scheduler.shutdown()
cap.release()
cv2.destroyAllWindows()


# In[12]:


print(len(face_data['right_eyes']),len(face_data['left_eyes']),len(face_data['mouth']))

