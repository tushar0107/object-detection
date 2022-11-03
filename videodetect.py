import cv2
import numpy as np
import time
import random

# Load the YOLO model
net = cv2.dnn.readNet('yolov3_training_last.weights','yolov3_testing.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Loading webcam or video
cap = cv2.VideoCapture("cars.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
colors = [(0,255,255),(255,0,255),(255,255,0),(255,255,155),(155,255,155),(155,255,255),(10,2255,200)]
color = random.choice(colors)

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            percent = str(round(confidence, 2)*100) + "%"
            cv2.rectangle(frame, (x-15, y-15), (x + w+15, y + h+15), color, 2)
            cv2.putText(frame, label + " " + percent, (x, y-5), font, 1/2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (40, 670), font, .7, (0, 255, 255), 1)
    cv2.putText(frame, "press [esc] to exit", (40, 690), font, .45, (0, 255, 255), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("Videocapturing stopped")
        break

cap.release()
cv2.destroyAllWindows()