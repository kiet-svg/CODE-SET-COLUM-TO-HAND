import cv2
import numpy as np
import mediapipe as mp
import time
import hand as hd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
timer = time.time()
deretor = hd.handDetector(detectionCon=0.7)

# Proper way to initialize audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

lst = volume.GetVolumeRange()
min = lst[0]
max = lst[1]

while True:
    ret, frame = cap.read()
    frame = deretor.findHands(frame)
    lmlist = deretor.findPosition(frame, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]

        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        dodai = math.hypot(x2 - x1, y2 - y1)
        print(dodai)

        if dodai < 25:
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        vol = np.interp(dodai, [25, 150], [min, max])
        volume.SetMasterVolumeLevel(vol, None)

    current_time = time.time()
    fps = 1 / (current_time - timer)
    timer = current_time
    cv2.putText(frame, f"FPS:{int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()