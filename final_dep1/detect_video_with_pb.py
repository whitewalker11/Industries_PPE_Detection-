import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2

import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils1 import draw_outputs
from gui_ppe import GUI_final
import os
import json

import csv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string('classes', 'data/coco.names', 'path to classes file')
flags.DEFINE_string('cam', 'data/cam.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('output', 'videos/detected/sample_airport_1.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')

try:
    with open(os.path.join(ROOT_DIR, "settings.json"), 'r') as config_buffer:
        config = json.loads(config_buffer.read())
except FileNotFoundError:
    print("setting not found")


def main(_argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    ################# GUI function call #####################

    model, cam = GUI_final()

    ########## Camera selection condition ###################

    if cam == " Unit1":
        c = config["cam"]["UNIT1_IP"]
        print(c)
    if cam == " Unit2":
        c = config["cam"]["UNIT2_IP"]
    if cam == " Unit3":
        c = config["cam"]["UNIT3_IP"]
    if cam == " Unit4":
        c = config["cam"]["UNIT4_IP"]
    if cam == " Unit5":
        c = config["cam"]["UNIT5_IP"]
    if cam == " Unit6":
        c = config["cam"]["UNIT6_IP"]

    model = tf.saved_model.load(config["model"]["maskmodel_path"])
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]

    logging.info('classes loaded')

    times = []

    ####### VIDEO CAP SELECTION CONDITION ###########
    if c == 0:
        vid = cv2.VideoCapture(c)
    else:
        vid = cv2.VideoCapture(f"{c}")

    out = None
    count = 0
    list_v = []

    ###### CAMERA COORDINATES #############
    y1 = config["cam_slice"]["y1"]
    y2 = config["cam_slice"]["y2"]
    x1 = config["cam_slice"]["x1"]
    x2 = config["cam_slice"]["x2"]

    while True:
        _, img = vid.read()
        if c == 0:
            img = img
        else:
            img = img[y1:y2, x1:x2]

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        outputs = infer(img_in)
        boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
            "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
        t2 = time.time()
        ti = (t2 - t1)

        count += 1
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, count, list_v, cam)
        img = cv2.putText(img, "Time: {:.2f}ms".format(1 / ti), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        cv2.imshow('output', img)

        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('output', 0) == -1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
