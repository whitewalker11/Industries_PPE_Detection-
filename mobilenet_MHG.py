import numpy as np
import argparse
import cv2
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
c=0




def detect_and_predict_gloves(frame,startX, startY, endX, endY, glovesnet):

    face = frame[startY:endY, startX:endX]
    cv2.imshow("gloves",face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = glovesnet.predict(face)

    return (preds)



def detect_and_predict_mask(frame,startX, startY, endX, endY, masknet):

    face = frame[startY:endY, startX:endX]
    cv2.imshow("mask",face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = masknet.predict(face)


    return (preds)



def detect_and_predict_helmet(frame,startX, startY, endX, endY, helmetnet):
    face = frame[startY:endY, startX:endX]
    cv2.imshow("helmet",face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = helmetnet.predict(face)

    return (preds)

def detect_and_predict_boots(frame,startX, startY, endX, endY, bootsnet):

    face = frame[startY:endY, startX:endX]
    cv2.imshow("B",face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = bootsnet.predict(face)


    return (preds)





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
'''ap.add_argument("-i", "--image", required=True,
	help="path to input image")'''

ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))




print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
helmetnet=load_model("PPE_helmet_2")
masknet=load_model("PPE_mask_20")
glovesnet=load_model("PPE_gloves_20")
bootsnet=load_model("boots_new_model")


vid_path="C:\\Users\\Dhruv\\Desktop\\TEST-VIDEO\\5.mp4"
vs=cv2.VideoCapture(vid_path)
# image = vs.read()



while 1:
    start = time.time()
    ret,image= vs.read()

    #image = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    (H, W) = image.shape[:2]
    print(h,w)
    #image=image[0:H,250:W]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    end = time.time()

        # fps
    t = end - start
    fps  = "Fps: {:.2f}".format(1 / t)
        # display a piece of text to the frame
    cv2.putText(image, fps, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



    for i in np.arange(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        color = (0, 255, 0)
        idx = int(detections[0, 0, i, 1])

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        startY = startY
        startX = startX



        if CLASSES[idx] == "person":
            #if endY > h:
                #endY = h - endY
            H=endY-startY
            h_height = int((H)/2)
            of = int(H/4)
            if H>400 and startX>250 and w-endX>250 and h-endY>10 and startY>100:




                preds = detect_and_predict_helmet(image, startX, startY-20, endX, abs(endY - h_height-of-35), helmetnet)
                for p_1 in zip(preds):
                    k = p_1[0]
                    (mask, withoutMask) = k

                    label = "helmet" if mask > withoutMask else "No helmet"
                    color1 = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                    global hi
                    if mask > withoutMask:
                        hi = 1
                    else:
                        hi = 0

                    label = "{}:".format(label)

                    cv2.putText(image, label, (startX + 10, startY + 30-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
                    cv2.circle(image, (startX + 5, startY + 25-60), radius=1, color=color1, thickness=3)




                preds1 = detect_and_predict_mask(image, startX+15, startY+20, endX-15, endY-h_height-of,masknet)
                for (pred1) in zip(preds1):
                    p = pred1[0]
                    (mask, withoutMask) = p

                    label = "mask" if mask > withoutMask else "No mask"
                    color1 = (0, 255, 0) if label == "mask" else (0, 0, 255)

                    global mi
                    if mask > withoutMask:
                        mi = 1
                    else:
                        mi = 0

                    label = "{}".format(label)

                    cv2.putText(image, label, (startX + 10, startY + 50-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)

                    cv2.circle(image, (startX + 5, startY + 45-60), radius=1, color=color1, thickness=3)




                (preds2) = detect_and_predict_gloves(image, startX-10, int(startY+of+20), endX+10, int(endY-of), glovesnet)
                for (pred2) in zip(preds2):
                    pr = pred2[0]
                    (gloves, withoutgloves) = pr

                    label = "gloves" if gloves > withoutgloves else "No gloves"
                    color = (0, 255, 0) if label=="gloves" else (0, 0, 255)

                    global gi
                    if gloves > withoutgloves:
                        gi = 1
                    else:
                        gi = 0

                    label = "{}".format(label)

                    cv2.putText(image, label, (startX + 10, startY + 70-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(image, (startX + 5, startY + 65 -60), radius=1, color=color, thickness=3)



                (preds3) = detect_and_predict_boots(image, startX, int(startY+h_height+of+10), endX, int(endY+20),bootsnet)
                for (pred3) in zip(preds3):
                    pr1 = pred3[0]
                    (boots,withoutboots) = pr1

                    label = "Boots" if boots > withoutboots else "No Boots"
                    color = (0, 255, 0) if label == "Boots" else (0, 0, 255)

                    global bi
                    if boots>withoutboots:
                        bi=1
                    else:
                        bi=0

                    print(label,bi)

                    v=max(boots, withoutboots) * 100

                    #label = "{}".format(label)

                    cv2.putText(image, label, (startX + 10, startY + 90-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(image, (startX + 5, startY + 85-60), radius=1, color=color, thickness=3)


                sum=mi+hi+gi+bi
                sum=4-sum
                if mi==0 or hi==0 or gi==0 or bi==0:
                    cv2.rectangle(image, (startX, startY - 60), (endX, endY), (80, 10, 10), 1)
                else:
                    cv2.rectangle(image, (startX, startY-60), (endX, endY), (80, 10, 10), 1)

                label = "{}-VIOLATION".format(sum)

                y = startY-60 - 15 if startY-60 - 15 > 15 else startY-60 + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 10, 10),4)

    sub_img = image[0:60, 0:W]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

    res = cv2.addWeighted(sub_img, 0.58, black_rect, 0.23, 2.0)

    image[0:60, 0:W] = res

    #cv2.rectangle(frame,(0,0),(455,40),(0, 0, 150), 2)
    #out.write(frame)
    cv2.putText(image, "PPE Kit Detection", (250, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)


    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.release()




# # show the output image
