from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.model = load_model('TLD_simulator.h5')

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

        prediction = self.model.predict(image)

        # {'unknown': 2, 'green': 0, 'yellow': 3, 'red': 1}, i.e. alphabetical


        # uint8 UNKNOWN=4
        # uint8 GREEN=2
        # uint8 YELLOW=1
        # uint8 RED=0

        if prediction==0:
          return TrafficLight.GREEN 
        elif prediction==1:
          return TrafficLight.RED 
        elif prediction==2:
          return TrafficLight.UNKNOWN 
        elif prediction==3
          return TrafficLight.YELLOW

        #################################################


        return TrafficLight.UNKNOWN
