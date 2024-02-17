import cv2
import time
import numpy as np
import HandTrackingModule2 as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
###################################
wcam = 640
hcam = 480
###################################
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ptime =0

detector = htm.HandDetector(detection_con=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
vol = 0
volbar = 400
volper = 0
while True:
    success, img = cap.read()
    img = detector.draw_hands(img)
    landmarks_list = detector.find_landmarks(img, draw=False)
    if len(landmarks_list) != 0:
        #print(landmarks_list[4], landmarks_list[8])
        x1,y1 = landmarks_list[4][1], landmarks_list[4][2]
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        #print(length)

        #Hand range: 25-220
        #Volume range: -65-0

        vol = np.interp(length,[25,220],[minvol,maxvol])
        volbar = np.interp(length,[25,220],[400,150])
        volper = np.interp(length, [25,220],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        if length < 25 :
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volbar)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f'{int(volper)}%',(40,450),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_PLAIN,
                2 , (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)