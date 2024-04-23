
"""

import cv2
import numpy as np


model_cfg = "C:\\Users\\Monster\\Desktop\\Custom_Yolov7\\yolov7-custom\\cfg\\training\\yolov7-custom.yaml"
model_weights = "C:\\Users\\Monster\\Desktop\\Custom_Yolov7\\yolov7-custom\\custom_yolov7.pt"
model_data = "C:\\Users\\Monster\\Desktop\\Custom_Yolov7\\yolov7-custom\\data\\custom_data.yaml"

net = cv2.dnn.readNet(model_cfg,model_weights)
with open(model_data, 'r') as f:
    classes = f.read().strip().split('\n')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()


    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                w = int(obj[2] * frame.shape[1])
                h = int(obj[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Real-time Object Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""




"""

import cv2
from darknet import load_net, load_meta, detect


model_cfg = "path/to/your/model.cfg"
model_weights = "path/to/your/model.weights"
model_data = "path/to/your/model.data"


net = load_net(model_cfg.encode("ascii"), model_weights.encode("ascii"), 0)
meta = load_meta(model_data.encode("ascii"))


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Kameradan bir kare alın

    detections = detect(net, meta, frame)


    for detection in detections:
        x, y, w, h = detection[2]  
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


    cv2.imshow('Real-time Object Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""



import torch
import cv2


model = torch.load("C:/Users/Monster/Desktop/Custom_Yolov7/yolov7-custom/custom_yolov7.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # result = model(frame)  # Modelinizi gerçek kareye uygulayın



    cv2.imshow('Real-time Object Detection', frame)



cap.release()
cv2.destroyAllWindows()

"""


import torch
import cv2
import numpy as np
from torchvision import transforms
from models.experimental import attemp_load

#model = torch.load("C:/Users/Monster/Desktop/Custom_Yolov7/yolov7-custom/custom_yolov7.pt")

model = attemp_load("C:/Users/Monster/Desktop/Custom_Yolov7/yolov7-custom/custom_yolov7.pt" , map Location = cpu)
#model = torch.load("C:/Users/Monster/Desktop/Custom_Yolov7/yolov7-custom/custom_yolov7.pt", map_location=torch.device('cpu'))['model']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # result = model(frame)  # Modelinizi gerçek kareye uygulayın



    cv2.imshow('Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""