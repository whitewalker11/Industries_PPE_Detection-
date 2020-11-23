import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from gui_ppe import GUI_final


flags.DEFINE_string('classes', 'data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('output', 'videos/detected/sample_airport_1.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


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




    model,cam=GUI_final()


    if cam=="Unit4":
        c=0
    else:
        c=0
    model = tf.saved_model.load('data/1_hand')
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    #vid = cv2.VideoCapture(f"{cam}")
    vid=cv2.VideoCapture(c)
    #print(f"{cam}")

    out = None
    count=0


    while True:
        _, img = vid.read()

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
        ti=(t2-t1)

        count+=1
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names,count,model)
        img = cv2.putText(img, "Time: {:.2f}ms".format(1/ti), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # if FLAGS.output:
            # out.write(img)
            # print('**writing**')
        cv2.imshow('output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
