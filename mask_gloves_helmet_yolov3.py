import math
from imutils.video import VideoStream
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
tt = tf.test.is_built_with_cuda()
print(tt)


def detect_and_predict_helmet(frame, startX, startY, endX, endY, masknet):
    face = frame[startY:endY, startX:endX]
    cv2.imshow("H", face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = masknet.predict(face)

    return (preds)


def detect_and_predict_mask(frame, startX, startY, endX, endY, masknet):
    face = frame[startY:endY, startX:endX]
    cv2.imshow("M", face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = masknet.predict(face)

    return (preds)


def detect_and_predict_gloves(frame, startX, startY, endX, endY, masknet):
    face = frame[startY:endY, startX:endX]
    cv2.imshow("G", face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = masknet.predict(face)

    return (preds)


confid = 0.5
thresh = 0.5

labelsPath = "MODELS/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "MODELS/yolov3.weights"
configPath = "MODELS/yolov3.cfg"
helmetnet = load_model("PPE_helmet_2")
masknet = load_model("PPE_mask_2")
glovesnet = load_model("PPE_Gloves_2")

###### use this for faster processing (caution: slighly lower accuracy) ###########

# weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
# configPath = "./yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR = 0

vid_path = "video/PPE1.avi"
vs = cv2.VideoCapture(vid_path)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output11.avi', fourcc, 20.0, (1000,700))

while True:
    start = time.time()
    (grab, frame) = vs.read()

    frame = cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_AREA)
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)
    # print(layerOutputs)

    end = time.time()

    # fps
    t = end - start
    fps = "Fps: {:.2f}".format(1 / t)
    # display a piece of text to the frame
    cv2.putText(frame, fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([abs(x), abs(y), int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

        status = []
        idf = idxs.flatten()

        for i in idf:

            if boxes[i][3] > 300 and H - (boxes[i][1] + boxes[i][3]) > 0:

                '''cv2.putText(frame, "Helmet", (1,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, "Mask", (1, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                cv2.putText(frame, "Gloves", (1, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)'''

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cen = [int(x + w / 2), int(y + h / 2)]
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                endX = x + w
                endY = y + h
                of = int(h / 4)
                hf = int(h / 2)
                tf = int(3 * h / 2)
                startX = abs(x - 10)
                startY = y - 10
                endX = endX + 10

                cv2.rectangle(frame, (startX, startY), (endX, endY), (30, 0, 0), 2)
                print(startX, startY, endX, endY)
                label = "{}-{}".format("Person", i + 1)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 0, 0), 2)

                preds = detect_and_predict_helmet(frame, startX, abs(startY - 10), endX, endY - hf - of, helmetnet)
                for p in zip(preds):
                    k = p[0]
                    (mask, withoutMask) = k

                    label = "helmet" if mask > withoutMask else "No helmet"
                    color1 = (0, 255, 0) if label == "helmet" else (0, 0, 255)

                    label = "{}: {:.0f}%".format(label, (max(mask, withoutMask) * 100) - 1)

                    cv2.putText(frame, label, (startX + 10, startY + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 1)
                    cv2.circle(frame, (startX + 5, startY + 10), radius=1, color=color1, thickness=4)

                preds1 = detect_and_predict_mask(frame, startX, startY + 20, endX, endY - hf - of, masknet)
                for (pred1) in zip(preds1):
                    p = pred1[0]
                    (mask, withoutMask) = p

                    label = "mask" if mask > withoutMask else "No mask"
                    color1 = (0, 255, 0) if label == "mask" else (0, 0, 255)
                    label = "{}: {:.0f}%".format(label, (max(mask, withoutMask) * 100) - 1)

                    cv2.putText(frame, label, (startX + 10, startY + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 1)

                    cv2.circle(frame, (startX + 5, startY + 25), radius=1, color=color1, thickness=4)

                (preds2) = detect_and_predict_gloves(frame, startX, int(startY + of + 20), endX, int(endY - of - 20),
                                                     glovesnet)
                for (pred2) in zip(preds2):
                    pr = pred2[0]
                    (gloves, no_gloves) = pr

                    label = "Gloves" if gloves > no_gloves else "No gloves"
                    color = (0, 255, 0) if label == "Gloves" else (0, 0, 255)
                    v = max(gloves, no_gloves) * 100

                    label = "{}: {:.0f}%".format(label, v - 1)

                    cv2.putText(frame, label, (startX + 10, startY + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.circle(frame, (startX + 5, startY + 40), radius=1, color=color, thickness=4)

    sub_img = frame[0:60, 0:W]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

    res = cv2.addWeighted(sub_img, 0.58, black_rect, 0.23, 2.0)

    frame[0:60, 0:W] = res

    # cv2.rectangle(frame,(0,0),(455,40),(0, 0, 150), 2)
    # out.write(frame)
    cv2.putText(frame, "PPE Kit Detection", (250, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imshow('P_P_E', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

vs.release()
# out.release()
cv2.destroyAllWindows()

