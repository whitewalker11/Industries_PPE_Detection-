from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import psycopg2
from datetime import date
import os
today = date.today()
from time import strftime

conn = psycopg2.connect(database=****, user=****, password=****, host=****, port=****)
# print ("Opened database successfully")
cur = conn.cursor()

########## S3 BUCKET ############
import boto3
from botocore.exceptions import NoCredentialsError

ACCESS_KEY = *******
SECRET_KEY = *******
try:
   s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                           aws_secret_access_key=SECRET_KEY)
except:
    print("no s3")


def upload_to_aws(local_file, bucket, s3_file):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def get_url(st):

    url = s3.generate_presigned_url('get_object',
                                    Params={
                                        'Bucket': '***',
                                        'Key': st,
                                    },
                                    ExpiresIn=3600)
    return url


YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names, count, list_v, cam,pro,img1):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    camera_unit = cam
    production_house = pro
    date = today.strftime("%m/%d/%y")
    im_id = today.strftime("%m%d%y")
    time = strftime("%H:%M:%S")
    im_id=str(im_id)+str(time)
    #print(im_id)

    for i in range(nums):

        if class_names[int(classes[i])] == 'no_mask':
            list_v.append(1)
            #if (objectness[i]) >= 0.20:
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

            x1 = x1y1[0]
            y1 = x1y1[1]
            x2 = x2y2[0]
            y2 = x2y2[1]

            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                              x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        else:
            list_v.append(0)
        if len(list_v) == 30:
            print(list_v)

            if sum(list_v) >= 15:
                flag = 1
                print(flag)
                t = flag

                class_v = "NO MASK"

                print(class_v, production_house,camera_unit, date, time)


                path = '/home/whitewalker/Documents/Projects/yolov3-tf2/PPE_exe/yolov3_tf2/data_img/'+str(im_id)+'.jpg'
                cv2.imwrite(path, img1)

                st="ppes_Data/"+str(im_id)+'.jpg'




                uploaded_status = upload_to_aws(path,'smartblast',st)
                print(uploaded_status)
                if uploaded_status is True:
                    #url=get_url(st)
                    url='****'+str(st)
                    print(url)


                    query = "INSERT INTO ppes (class,production_house,camera_unit,date,time,image_url) VALUES (%s, %s, %s, %s, %s ,%s);"
                    data = (class_v, production_house, camera_unit, date, time,url)

                    cur.execute(query, data)
                    conn.commit()





            else:
                flag = 0
                print(flag)
                t = flag

            list_v.clear()


    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
