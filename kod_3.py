from PIL import Image
import numpy as np
import cv2
import os
import time

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
black = [0, 0, 0]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0)

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    lowerLimit = np.array([0,0,0],dtype=np.uint8)
    upperLimit = np.array([180,20,50],dtype=np.uint8)
    return lowerLimit, upperLimit

while True:
    ret, frame = cap.read()

    color = (200,0,0)
    cv2.line(frame,(128,40),(128,440),color,2)
    cv2.line(frame,(256,40),(256,440),color,2)
    cv2.line(frame,(384,40),(384,440),color,2)
    cv2.line(frame,(512,40),(512,440),color,2)
    cv2.line(frame,(640,40),(640,440),color,2)

    cv2.line(frame,(0,40),(640,40),color,2)
    cv2.line(frame,(0,440),(640,440),color,2)



    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=black)
    mask_black = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    mask_red = cv2.inRange(hsvImage,lower_red,upper_red)

    mask_ = Image.fromarray(mask_black)
    bbox = mask_.getbbox()

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Her bir konturu dolaş
    for contour in contours:
        # Konturun alanını hesapla
        area = cv2.contourArea(contour)

        # Eğer alan belirli bir değerden büyükse (örneğin, gürültüyü filtrelemek için)
        if area > 100:
            # Konturun etrafına bir dikdörtgen çiz
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Red Circle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if(x1 < 128 and x2 < 128):
            print("L")
        if(x1 >= 128 and x1 <256 and x2>= 128 and x2 < 256):
            print("ML")
        if (x1 >= 256 and x1 < 384 and x2 >= 256 and x2 < 384):
            print("M")
        if (x1 >= 384 and x1 < 512 and x2 >= 384 and x2 < 512):
            print("MR")
        if (x1 >= 512 and x1 < 640 and x2 >= 512 and x2 < 640):
            print("R")

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        #print(bbox)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()