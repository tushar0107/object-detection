import cv2
import numpy as np
import random

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("yolov3_training.weights", "yolov3_testing.cfg")

#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#get layers of the network
layer_names = net.getLayerNames()

#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

# Capture frame-by-frame
img = cv2.imread("img7.jpeg")
img = cv2.resize(img, None, fx=1, fy=1)
cv2.imshow("Image",img)
cv2.waitKey(2000)

height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    #Detecting objects
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
colors = [(0,255,255),(255,0,255),(155,255,155),(155,255,255),(10,2255,200),(0,255,0),(255,0,0),(0,0,255)]


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        color = random.choice(colors)
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        percent = str(round(confidence, 2)*100) + "%"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + percent, (x, y-5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)
        print(label, percent)

cv2.imshow("Image",img)
cv2.waitKey(5000)