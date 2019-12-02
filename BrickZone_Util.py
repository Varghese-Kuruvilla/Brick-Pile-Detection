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
        self.detections = None #Detections from darknet 
        self.frame_rgb = None  #RGB frame
        self.cropped_image = None #Image cropped to BB dimensions
        self.color_code = None
        self.bb_confidence_ls = [] #List indicating if the centroid of the BB lies within/outside the centre of the image. 1-Centroid lies within 60% of the image centre. 0- Centroid lies outside 60% of the image centre
        self.brick_pile_color = {1:"Red",2:"Blue",3:"Green",4:"Orange"}
        self.dict_class_names = {b'Brick stack':0,b'Construction zone':1}  #Dict for coding class names
        self.dict_lowconf_pos = {0:[] , 1:[], 2:[], 3:[], 4:[]} #Dict containing values as lists for averaging local positions obtained over 5 consecutive frames
        self.dict_returned_pos = {0:0, 1:0, 2:0, 3:0, 4:0} #Dict indicating if local positions of the construction zone and brick stacks have been indentified

        

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
        #rospy.init_node('listener', anonymous=True) 


    def select_roi(self):
        '''Function to crop the image to the BB dimensions and find out colour of the brick stack within the ROI
        Returns: None'''
        # winName_crop = "Cropped Image"
        # cv2.namedWindow(winName_crop,cv2.WINDOW_NORMAL)
        self.color_code = None
        for detection in self.detections: 
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(float(x), float(y), float(w), float(h))

            xmin,ymin,xmax,ymax = self.check_crop_bound(xmin,ymin,xmax,ymax)  #Check if the image is cropped correctly

            self.cropped_image = self.frame_rgb[ymin:ymax,xmin:xmax]
            # cv2.imshow(winName_crop,self.cropped_image)
            # key = cv2.waitKey(0)
            # if(key & 0xFF == ord('q')):
            #     cv2.destroyAllWindows()
            #     exit() 

            #Converting self.cropped_image into HSV scale and extracting Hue channel alone
            
            hsv_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2HSV)

            self.color_code = color_obj.label(hsv_image)  #Brick_pile_color is a string that gives us the color of the brick


    def check_crop_bound(self,xmin,ymin,xmax,ymax):
        '''Function to check if coordinates for cropping are within image bounds
        Returns: coordinates xmin,ymin,xmax,ymax after'''
        coord_ls = [xmin,ymin,xmax,ymax]
        for i in range(0,4):
            if (coord_ls[i] <= 0):
                coord_ls[i] = 0

            if(i == 0 or i == 2):
                if(coord_ls[i] >= self.frame_rgb.shape[1]):
                    coord_ls[i] = self.frame_rgb.shape[1]

            elif(i == 1 or i == 3):
                if(coord_ls[i] >= self.frame_rgb.shape[0]):
                    coord_ls[i] = self.frame_rgb.shape[0]

        return coord_ls

    #Both of these are helper functions used for drawing boxes on the objects
    def convertBack(self, x, y, w, h):
        
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self ,img, detections,class_label):
        '''Function to draw boxes around the ROI 
        class_label indicates colour of the brick stack
        or construction zone'''
        for detection in detections:
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
                        " [" + str(round(detection[1] * 100, 2)) + "]" + 
                        str(class_label),
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            
            display(img,"Inference")
            #print("self.brick_pile_color[color_code]:",self.brick_pile_color[self.color_code])

            #cv2.putText(img,
            #            #detection[0].decode() +
            #            self.brick_pile_color[self.color_code],
            #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #            [0, 255, 0], 2)
        #return img

    def subscribe_position(self):
        '''Subscribes to the local_position topic'''
        drone_pos = rospy.wait_for_message('/dji_sdk/local_position',PointStamped)
        x_pos = float(drone_pos.point.x)
        y_pos = float(drone_pos.point.y)
        z_pos = float(drone_pos.point.z)

        position = [x_pos,y_pos,z_pos] #Position contains the x,y,z coordinates of the drone

        return position

    def chk_bb_confidence(self):
        '''Checks if the BBs in self.detections lies within 60% of the centre of the image
        Populates list self.bb_confidence_ls with a value 1 if the BB lies within 60% of the image centre 
        and a value 0 if it lies outside the 60% of the image centre'''

        #Clear self.bb_confidence_ls
        self.bb_confidence_ls.clear()

        for detection in self.detections:
            #Obtaining x,y,w,h values from self.detections and converting these values into xmin,ymin,xmax,ymax format
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))

            #Centroid of each BB 
            #Format:[x_centroid,y_centroid]
            bb_centroid = [int((xmin + xmax)/2),int((ymin+ymax)/2)]
            
            if(((bb_centroid[0] >= 0.20 * self.frame_rgb.shape[1]) and (bb_centroid[0] <= 0.80 * self.frame_rgb.shape[1])) and ((bb_centroid[1] >= 0.20 * self.frame_rgb.shape[0]) and (bb_centroid[1] <= 0.80 * self.frame_rgb.shape[0]))):
                self.bb_confidence_ls.append(1) #Centroid of the BB lies in the centre of the image
            else:
                self.bb_confidence_ls.append(0) #Centroid of the BB lies outside image centre


    def infer(self, in_image, thresh = 0.5):
        ''' 
            Forward pass of the model for the given RGB input image. 
            Returns: (Class_ID, Local_Position)
                        Class_ID : An integer which maps to the enum representing construction zone and bricks
                        {0: "Construction Zone"; 1: "Red Brick"; 2: "Blue Brick"; 3: "Green Brick"; 4: "Orange Brick"}
                        Local_Position: A float representing the local position of the drone, this is obtained from 
                                        subscribing to one of the ROS topics
        '''

        #Initialize list to return a list of tuples of the form (color,local_position)
        return_ls = []
        #self.frame_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
        self.frame_rgb = in_image
        frame_resized = self.frame_rgb
        #frame_resized = cv2.resize(self.frame_rgb, (self.net_width, self.net_height), interpolation=cv2.INTER_LINEAR) #changed
        # Copy image to darknet format 
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        # Forward pass 
        self.detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh)
        
        #Function to check if the centroid of the BB is within 60% of the image centre or outside 60% of the image centre
        self.chk_bb_confidence()

        if not self.detections:
            return []
        # Obtain the best detection from the list of detections
        for (i,detection) in enumerate(self.detections):
            #best_detection = self.detections[0]
            #best_detection = detection[0]
            best_detection = detection
            #### Check the class_ID of the detection and based on the ID return the local_position if 
            #### its a construction zone or do further processing if its a brick

            class_code = self.dict_class_names[best_detection[0]] 

            if(class_code == 1):  #Condition for construction zone detection
                if(self.dict_returned_pos[0] == 1): #Local position has already been returned
                    continue 
                self.color_code = 0
                self.cvDrawBoxes(frame_resized,[best_detection],'0') #0 for the construction zone
                #local_position = self.subscribe_position()
                #return (0,local_position)
                
                    
            elif (class_code == 0): #Condition for brick stack detection
                # crop the image and check the color of the image 
                self.select_roi()
                if(self.color_code!= None and self.dict_returned_pos[self.color_code] == 1): #Local position of the brick stack has been returned
                    continue

                self.cvDrawBoxes(frame_resized,[best_detection],self.color_code)
                #local_position = self.subscribe_position() #changed
                #return (self.color_code, local_position) #changed

            if(self.bb_confidence_ls[i-1] == 1): #Centroid lies within 60% of the image centre
                self.dict_returned_pos[self.color_code] = 1
                #local_position = self.subscribe_position()
                #return_ls.append((self.color_code, local_position)) #List containing tuples of (self.color_code,local positions) to be returned
                return_ls.append(self.color_code)

            elif(self.bb_confidence_ls[i-1] == 0): #Centroid lies outside 60% of the image centre
                #local_position = self.subscribe_position()
                #self.dict_lowconf_pos[self.color_code].append(local_position)
                self.dict_lowconf_pos[self.color_code].append(self.color_code) #Append local_position to the corresponding list in the dictionary
                print("self.dict_lowconf_pos",self.dict_lowconf_pos)

                if(len(self.dict_lowconf_pos[self.color_code]) == 5): #After 5 detections we average the local positions and return them
                    self.dict_returned_pos[self.color_code] = 1
                    #return_ls.append(self.color_code, local_position)
                    return_ls.append(self.color_code)
            print("Self.dict_returned_pos:",self.dict_returned_pos)
        return return_ls

#Functions for debug and testing---------------------------------------------------------------
def display(img,txt):
    winName = txt
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)
    #out.write(img)
    cv2.imshow(winName,img)
    cv2.waitKey(1)
    #key = cv2.waitKey(0)
    #if(key & 0xFF == ord('q')):
    #    cv2.destroyAllWindows()
    #    exit()

def subscribe_image():
    '''Function to subscribe to the raw image and convert it into an opencv image
    Returns: Opencv image'''
    image_msg = rospy.wait_for_message('/camera/color/image_raw',Image)
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg,"bgr8")
    return cv_image

if __name__ == '__main__':
    #Initialize VideoWriter for testing 
    out = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (640, 480))

    #File for writing local position output
    f = open("test.txt","w")
    
    brickzone_obj = BrickZone() #Object of class BrickZone
    color_obj = colorlabel() #Object of the class colorlabel
    brickzone_obj.initialize_network("/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick.cfg","/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick_2000.weights","/home/varghese/challenge_2/brick_train/brick_stack_construction_zone_v1/Weights/brick.data") #Initializing network

    cap = cv2.VideoCapture("/home/varghese/brick_data/data_nov21/cropped_videos/rosbag_3_cropped.m4v")
    ret = True
    while(cap.isOpened() and ret == True):
    #while(not(rospy.is_shutdown())):
        #frame = subscribe_image() #Subscribing to the RGB Image
        ret, frame = cap.read()
        display(frame,"Live feed")
        ret_val = brickzone_obj.infer(frame) #ret_val is a tuple (color_code,local_position)
        f.write(str(ret_val))
        #print("Ret val:",ret_val)
