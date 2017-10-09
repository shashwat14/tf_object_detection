#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import tensorflow as tf
import sys
sys.path.append('/media/vision/New Volume/models/research/object_detection')
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util
import message_filters
from geometry_msgs.msg import Twist
class ObjectDetectionTFNode:

    def __init__(self):

        self.bridge = CvBridge()
        self._model()
        self._subscribers()
        self._publishers()
        print "Init finished"

    def _subscribers(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as self.sess:
                img_sub1 = message_filters.Subscriber('/camera1/usb_cam1/image_raw', Image) #Change
                img_sub2 = message_filters.Subscriber('/camera2/usb_cam2/image_raw', Image) #Change
                img_sub3 = message_filters.Subscriber('/camera3/usb_cam3/image_raw' ,Image) #Change
                self.ts = message_filters.ApproximateTimeSynchronizer([img_sub1, img_sub2, img_sub3], 10, 1., allow_headerless=True) #Change
                self.ts.registerCallback(self.imageCallback)

    def _publishers(self):
        self.image_publisher = rospy.Publisher('/iMiev/tf_object_detection', Image, tcp_nodelay=True, queue_size=1) #Change

    def _model(self):

        MODEL_NAME = '/home/vision/catkin_ws/src/tf_object_detection/scripts/model' #Change
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #Change
        PATH_TO_LABELS = '/media/vision/New Volume/models/research/object_detection/data/mscoco_label_map.pbtxt' #Change
        NUM_CLASSES = 90
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def imageCallback(self, img1, img2, img3): #Change
        try:
            image_np, image_np_expanded = self.getImageTensors(img1)
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor:image_np_expanded})
            transformedImage = self.transformImage(image_np, boxes, scores, classes, num)

            image_np, image_np_expanded = self.getImageTensors(img2)
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor:image_np_expanded})
            transformedImage = self.transformImage(image_np, boxes, scores, classes, num)

            image_np, image_np_expanded = self.getImageTensors(img3)
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor:image_np_expanded})
            transformedImage = self.transformImage(image_np, boxes, scores, classes, num)

            image = self.bridge.cv2_to_imgmsg(transformedImage, 'rgb8')
            self.image_publisher.publish(image)

        except CvBridgeError as e:
            print e

    def getImageTensors(self, img):
        image = self.bridge.imgmsg_to_cv2(img, 'rgb8')
        image_np = image.astype(np.uint8)
        image_np_expanded = np.expand_dims(image, axis=0)
        return image_np, image_np_expanded

    def transformImage(self, image_np, boxes, scores, classes, num):
        image = image_np.reshape(480, 640, 3)
        for i in range(boxes.shape[1]):
            class_name = self.category_index[classes[0,i]]['name']
            if class_name != 'person':
                continue
            if scores[0,i] > 0.2: #Confidence
                x1, y1, x2, y2 = boxes[0,i]
                x1 = int(x1 * 480)
                x2 = int(x2 * 480)
                y1 = int(y1 * 640)
                y2 = int(y2 * 640)
                image[x1:x2, y1:y2] = 0
        return image

if __name__ == '__main__':

    rospy.init_node('ObjectDetectionTFNode', anonymous=True)
    o = ObjectDetectionTFNode()
    rospy.spin()
