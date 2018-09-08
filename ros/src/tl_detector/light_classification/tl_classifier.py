from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np
import cv2
import rospy
import tensorflow as tf


class TLClassifier(object):
    def __init__(self):
        self.model = load_model('tl_detection_model/TLD_simulator.h5')
        self.graph = tf.get_default_graph()
        self.model._make_predict_function()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


        ##################################################
      
        # TODO:  Make sure that "image" is in the correct format expected by predict (should it be in a list, RGB vs BGR, etc.)
        test_image = cv2.resize(image, (224, 224)) 
        test_image = np.array(test_image)
        test_image = test_image[...,::-1]
        test_image = test_image.astype('float32')
        test_image /= 255.
        test_image = np.expand_dims(test_image,0)
        with self.graph.as_default():
          prediction = self.model.predict(test_image)[0]

        label = np.argmax(prediction)
        rospy.loginfo("{}".format(label))

        # {'unknown': 2, 'green': 0, 'yellow': 3, 'red': 1}, i.e. alphabetical

        # uint8 UNKNOWN=4
        # uint8 GREEN=2
        # uint8 YELLOW=1
        # uint8 RED=0

        if label==0:
          return TrafficLight.GREEN 
        elif label==1:
          return TrafficLight.RED 
        elif label==2:
          return TrafficLight.UNKNOWN 
        elif label==3:
          return TrafficLight.YELLOW

        #################################################


        return TrafficLight.UNKNOWN
