import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
from ctypes import *
import math
import os
import cv2
import numpy as np
import time
import darknet

#ROS Dependencies
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

#For colour determination
from color_label import colorlabel

#Global variables for debugging
detected = 0
not_detected = 0

class read_inference:
    def __init__(self):
        #Data members for reading the inference from darknet
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.configPath = None
        self.weightPath = None
        self.metaPath = None
        self.darknet_image = None

        self.detections = None #BB dimensions
        self.frame_rgb = None  #RGB frame
        self.cropped_image = None #Image cropped to BB dimensions
        self.color_code = None 
        self.brick_pile_color = {1:"Red",2:"Blue",3:"Green",4:"Orange"}

    def initialize_network(self):  #Function to initialize darknet parameters 
        self.configPath = "/home/varghese/challenge_2/brick_train/brick_pile_train_v2/Weights/brick.cfg"
        self.weightPath = "/home/varghese/challenge_2/brick_train/brick_pile_train_v2/Weights/brick_4000.weights"
        self.metaPath = "/home/varghese/challenge_2/brick_train/brick_pile_train_v2/Weights/brick.data"
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
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


        #self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
        #                                darknet.network_height(self.netMain),3)
        self.darknet_image = darknet.make_image(1280,720,3) 


    def inference(self,frame_rgb): #Function for inference i.e returning the bounding box 
        self.frame_rgb = frame_rgb 
       # self.frame_resized = cv2.resize(frame_rgb,
       #                            (darknet.network_width(self.netMain),
       #                             darknet.network_height(self.netMain)),
       #                            interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image,self.frame_rgb.tobytes())

        self.detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        print("self.detections:",self.detections) 

        self.select_roi()   #Function to crop the image to BB dimensions

        #Debug Block - For counting the number of detections
        #---------------------------------------------------------------------#
        if(len(self.detections) == 0):
            global not_detected
            not_detected = not_detected + 1
        else:
            global detected
            detected = detected + 1
        
        image = self.cvDrawBoxes(self.frame_rgb)
        cv2.imshow('Demo', image)
        cv2.waitKey(1)

        #----------------------------------------------------------------------#
        

    def select_roi(self):                #Function to crop the image to BB dimensions
        winName_crop = "Cropped Image"
        cv2.namedWindow(winName_crop,cv2.WINDOW_NORMAL)
        for detection in self.detections: 
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(float(x), float(y), float(w), float(h))

            self.cropped_image = self.frame_rgb[ymin:ymax,xmin:xmax]
            cv2.imshow(winName_crop,self.cropped_image)
            key = cv2.waitKey(0)
            if(key & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                exit() 

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


if __name__ == "__main__":

    cap = cv2.VideoCapture("/home/varghese/brick_data/data_5th_nov/capture_1.avi")
    color_obj = colorlabel()    #Object of the colorlabel class
    inf_obj = read_inference() #Create object inf_obj of class read_inference
    inf_obj.initialize_network()  #Initialize network
    ret = True
    while(cap.isOpened() and ret==True):
        ret, frame_read = cap.read()
        inf_obj.inference(frame_read)   #Read the inference

        #For debug
        if(detected + not_detected == 40): 
            print("Number of frames detected:",detected)
            print("Number of frames missed:",not_detected)

