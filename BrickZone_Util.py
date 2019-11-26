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

#ROS Dependencies
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import PointStamped

#For colour determination
from color_label import colorlabel

#Global variables for debugging
detected = 0
not_detected = 0

class BrickZone(object):
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
        self.darknet_image = darknet.make_image(640,480,3) #Changed

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
        rospy.init_node('listener', anonymous=True)


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
        
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self ,img):
        
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

    def subscribe_position(self):
        drone_pos = rospy.wait_for_message('/dji_sdk/local_position',PointStamped)
        x_pos = float(drone_pos.point.x)
        y_pos = float(drone_pos.point.y)
        z_pos = float(drone_pos.point.z)

        position = [x_pos,y_pos,z_pos] #Position contains the x,y,z coordinates of the drone

        return position


    def infer(self, in_image, thresh = 0.5):
        ''' 
            Forward pass of the model for the given RGB input image. 
            Returns: (Class_ID, Local_Position)
                        Class_ID : An integer which maps to the enum representing construction zone and bricks
                        {0: "Construction Zone"; 1: "Red Brick"; 2: "Blue Brick"; 3: "Green Brick"; 4: "Orange Brick"}
                        Local_Position: A float representing the local position of the drone, this is obtained from 
                                        subscribing to one of the ROS topics
        '''

        self.frame_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
        frame_resized = self.frame_rgb
        #frame_resized = cv2.resize(self.frame_rgb, (self.net_width, self.net_height), interpolation=cv2.INTER_LINEAR) #changed
        # Copy image to darknet format 
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        # Forward pass 
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh)
        print("detections:",detections)

        if not detections:
            return []
        # Obtain the best detection from the list of detections 
        best_detection = detections[0]

        #### Check the class_ID of the detection and based on the ID return the local_position if 
        #### its a construction zone or do further processing if its a brick

        if (best_detection[0] == 1):
            return 0
            #local_position = self.subscribe_position() #changed
            #return (0, local_position) #changed

        elif (best_detection[0] == 0):
            self.detections = best_detection
            # crop the image and check the color of the image 
            self.select_roi()
            #local_position = self.subscribe_position() #changed
            #return (self.color_code, local_position) #changed
            return self.color_code

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
    brickzone_obj.initialize_network("/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick.cfg","/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick_2000.weights","/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick.data") #Initializing network

    cap = cv2.VideoCapture("/home/varghese/brick_data/data_nov21/cropped_videos/rosbag_3_cropped.m4v")
    ret = True
    while(cap.isOpened() and ret == True):
        ret, frame = cap.read()
        print("frame.shape",frame.shape)
        display(frame,"Live feed")
        ret_val = brickzone_obj.infer(frame)
        print("Ret val:",ret_val)
