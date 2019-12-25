import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
#sys.path.append(0, "/opt/ros/kinetic/lib/python2.7/dist-packages")
from ctypes import *
import math
import os

import numpy as np
import time
import darknet
import glob

#ROS Dependencies
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import PointStamped

#For colour determination
#from color_label import colorlabel

#Global variables for debugging
detected = 0
not_detected = 0

class BrickZone(object):
    '''
    Class for detecting UAV,UGV construction zone and brick pick up zones
    '''

    def __init__(self):
        super(BrickZone, self).__init__()

        self.cfg_path = None
        self.weights_path = None
        self.meta_path = None
        self.netMain = None
        self.metaMain = None
        self.net_width = None
        self.net_height = None
        self.net_channels = 3

        self.altNames = None
        self.detections = None #BB dimensions
        self.frame_rgb = None  #RGB frame
        self.cropped_image = None #Image cropped to BB dimensions
        self.color_code = None 
        self.brick_pile_color = {1:"Red",2:"Blue",3:"Green",4:"Orange"}

        
        self.uav_constr_detections = [] #List containing detections for UAV construction zone
        self.uav_constr_pos_ls = []
        self.uav_constr_pos_conf = []
    def initialize_network(self, cfg_path, weights_path, meta_path):

        if not os.path.exists(cfg_path):
            raise ValueError("Invalid config path `" +
                                os.path.abspath(cfg_path)+"`")
        else:
            self.cfg_path = cfg_path
        if not os.path.exists(weights_path):
            raise ValueError("Invalid weight path `" +
                                os.path.abspath(weights_path)+"`")
        else:
            self.weights_path = weights_path
        if not os.path.exists(meta_path):
            raise ValueError("Invalid data file path `" +
                                os.path.abspath(meta_path)+"`")
        else:
            self.meta_path = meta_path	

        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.cfg_path.encode("ascii"), self.weights_path.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.meta_path.encode("ascii"))

        self.net_width = darknet.network_width(self.netMain)
        self.net_height = darknet.network_height(self.netMain)
        #self.darknet_image = darknet.make_image(self.net_width, self.net_height, self.net_channels)
        self.darknet_image = darknet.make_image(1920,1080,3) #Changed

        if self.altNames is None:
            try:
                with open(self.meta_path) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        #Initializing ros Node
        #rospy.init_node('listener', anonymous=True)


    def select_roi(self):                #Function to crop the image to BB dimensions
        # winName_crop = "Cropped Image"
        # cv2.namedWindow(winName_crop,cv2.WINDOW_NORMAL)
        for detection in self.detections: 
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(float(x), float(y), float(w), float(h))

            self.cropped_image = self.frame_rgb[ymin:ymax,xmin:xmax]
            # cv2.imshow(winName_crop,self.cropped_image)
            # key = cv2.waitKey(0)
            # if(key & 0xFF == ord('q')):
            #     cv2.destroyAllWindows()
            #     exit() 

            #Converting self.cropped_image into HSV scale and extracting Hue channel alone
            
            hsv_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2HSV)

            self.color_code = color_obj.label(hsv_image)  #Brick_pile_color is a string that gives us the color of the brick

    #Both of these are helper functions used for drawing boxes on the objects
    def convertBack(self, x, y, w, h):
        '''
        Converts yolo annotation format to pascal VOC format

        Parameters

        -------------

        x,y,w,h:float
                x,y: absolute x and y coordinates of the image centre
                w,h: absolute width and height of the BB

        -------------

        Returns

        -------------

        xmin,ymin,xmax,ymax:int
                            xmin,ymin: x and y coordinates of the top left corner of the BB
                            xmax,ymax: x and y coordinates of the bottom right corner of the BB
        '''
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self ,img):
        '''
        Function to draw rectangles around the detected classes

        Parameters

        ---------------

        img:numpy Array
            Input Image

        Returns

        --------------

        img: numpy Array
            Image with rectangles draw around the detected classes
        '''
        for detection in self.detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            #cv2.putText(img,
            #            detection[0].decode() +
            #            " [" + str(round(detection[1] * 100, 2)) + "]",
            #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #            [0, 255, 0], 2)
            print("self.brick_pile_color[color_code]:",self.brick_pile_color[self.color_code])

            cv2.putText(img,
                        #detection[0].decode() +
                        self.brick_pile_color[self.color_code],
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img



    def cvDrawBoxes_test(self ,img):
        
        for detection in self.detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img

    def subscribe_position(self):
        '''
            Subscribes to the local position topic
            Returns

            ---------------

            position: list
                    Contains x,y,z coordinates indicating the drones current position
        '''
        drone_pos = rospy.wait_for_message('/dji_sdk/local_position',PointStamped)
        x_pos = float(drone_pos.point.x)
        y_pos = float(drone_pos.point.y)
        z_pos = float(drone_pos.point.z)

        position = [x_pos,y_pos,z_pos] 

        return position


    def find_bb_centroid(self,detections):
        '''
            Computes centroids of the BBs based on the obj_code
            
            Parameters
            __________

            detections: list
                        Contains detections in the following format:  [(b'class_name', confidence score, (centre_x, centre_y, absolute_BB_width, absolute_BB_height))...]
            Returns
            __________

            centroid_ls: list
                        [[centroid_xcoordinate,centroid_ycoordinate]]
                        
        '''
        
        centroid_arr = np.zeros((1,2))
        print("detections:",detections)                    
        for detection in detections:
            centroid_xcoor,centroid_ycoor = detection[2][0],detection[2][1]
            centroid_arr = centroid_arr + (centroid_xcoor,centroid_ycoor)

        centroid_arr = centroid_arr // len(detections)
        centroid_ls = centroid_arr.tolist()
        return centroid_ls

    def check_obj_location(self,centroid_ls):
        '''
            Checks if the centroid lies within 60% of the image centre

            Parameters

            ------------
            
            centroid_ls: list
                        [[centroid_xcoordinate,centroid_ycoordinate]]
            Returns

            ------------

            bool
                True if the centroid lies within 60% of the image centre , else False
        '''
        print("centroid_ls:",centroid_ls)
        #Error checking: Check if the centroid is out of image bounds and make required corrections
        for coord in centroid_ls:
            if(coord[0] <=0):
                coord[0] = 0
            elif(coord[0] >= self.frame_rgb.shape[1]):
                coord[0] = self.frame_rgb.shape[1]

            if(coord[1] <= 0):
                coord[1] = 0
            elif(coord[1] >= self.frame_rgb.shape[0]):
                coord[1] = self.frame_rgb.shape[0]


        #Check if centroid lies within 60% of the image centre

        for coord in centroid_ls:
            if((coord[0] >= (0.20 * self.frame_rgb.shape[1])) and (coord[0] <= (0.80 * self.frame_rgb.shape[1])) and (coord[1] >= 0.20 * self.frame_rgb.shape[0]) and (coord[1] <= 0.80 * (self.frame_rgb.shape[0]))):
                    return True

            else:
                
                    return False

            
                

    def infer(self, in_image, thresh = 0.5):
        ''' 
            Forward pass of the model for the given RGB input image.

            Parameters
            -----------------
            in_image:numpy array 
                    Input RGB image
        '''


        #Local Variables
        uav_constr_detections = []
        centroid_ls = []
        local_pos_ls = []

        self.frame_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
        frame_resized = self.frame_rgb
        # Copy image to darknet format 
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        # Forward pass 
        self.detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh)
        print("self.detections:",self.detections)

        #Drawing detections on the image
        test_img = self.cvDrawBoxes_test(frame_resized)
        test_img = cv2.cvtColor(test_img,cv2.COLOR_RGB2BGR)
        cv2.imshow("Detections on the test image",test_img)
        key = cv2.waitKey(0)
        if(key & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            exit()

        if not self.detections:
            return []

        else:
            for detection in self.detections:
                if(detection[0] == b'channel' and int(detection[1]*100)>=60):
                    uav_constr_detections.append(detection)

                #TODO: Conditions for rest of the classes to be added once the detector is trained for them
            
            if(len(uav_constr_detections)>=2):
                centroid_ls = self.find_bb_centroid(uav_constr_detections)
                ret = self.check_obj_location(centroid_ls)
                
                #local_pos_ls = self.subscribe_position()
                self.uav_constr_pos_ls.append(local_pos_ls)

                #Ret = True indicates that obj centroid is located within 60% of the image centre
                if(ret == True):
                    self.uav_constr_pos_conf.append(1) #High confidence
                elif(ret == False):
                    self.uav_constr_pos_conf.append(0) #Low confidence
        

        
        print("self.uav_constr_pos_conf:",self.uav_constr_pos_conf)

            

#Functions for debug
def display(img,txt):
    winName = txt
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)
    cv2.imshow(winName,img)
    cv2.waitKey(1)
    #key = cv2.waitKey(0)
    #if(key & 0xFF == ord('q')):
    #    cv2.destroyAllWindows()
    #    exit()

if __name__ == '__main__':
    brickzone_obj = BrickZone() #Object of class BrickZone
    brickzone_obj.initialize_network("/home/varghese/challenge_2/brick_train/servoing_ver_3/python_scripts/brick.cfg","/home/varghese/challenge_2/brick_train/servoing_ver_3/weights/yolov3-tiny_best.weights","/home/varghese/challenge_2/brick_train/servoing_ver_3/python_scripts/brick.data") #Initializing network

    #cap = cv2.VideoCapture("/home/varghese/brick_data/data_nov21/cropped_videos/rosbag_3_cropped.m4v")

    #ret = True
    #while(cap.isOpened() and ret == True):
    #    ret, frame = cap.read()
    #    print("frame.shape",frame.shape)
    #    display(frame,"Live feed")
    #    ret_val = brickzone_obj.infer(frame)
    #    print("Ret val:",ret_val)


    for img_path in glob.glob("/home/varghese/challenge_2/brick_train/servoing_ver_3/test_detector/*.jpg"):
        frame = cv2.imread(img_path)
        ret_val = brickzone_obj.infer(frame)


