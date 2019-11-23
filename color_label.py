#Script for determining colour within the bounding box
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import pickle

class colorlabel:
    def __init__(self):
        self.hue_value = None
        self.sat_value = None
        self.v_value = None

        #Variables for debugging
        self.count = 0 
        self.color_ls = []

    def label(self,image):

        hue_channel = image[:,:,0] #Hue channel
        hist = cv2.calcHist([hue_channel],[0],None,[180],[0,179])
        self.hue_value = np.where(hist == np.amax(hist))
        self.hue_value = self.hue_value[0][0]
        #print("self.hue_value:",self.hue_value)


        #Histogram of the S channel
        s_channel = image[:,:,1]
        hist_sat = cv2.calcHist([s_channel],[0],None,[256],[0,256]) 
        self.sat_value = np.where(hist_sat == np.amax(hist_sat))
        #print("self.sat_value:", self.sat_value[0][0])


        #Histogram of the V Channel
        v_channel = image[:,:,2]
        hist_v = cv2.calcHist([v_channel],[0],None,[256],[0,256])
        self.v_value = np.where(hist_v == np.amax(hist_v))
        #print("self.v_value:",self.v_value[0][0])



        #For debugging-----------------------------------------------------------------
        self.color_ls.append([self.hue_value,self.sat_value[0][0],self.v_value[0][0]])
        self.count = self.count + 1
        if(self.count == 30):
            with open("red_brick.pkl","wb") as f:
                pickle.dump(self.color_ls,f)
                inp = input("Done writing into pickle file")
        #-------------------------------------------------------------------------------#

        


        if((self.hue_value>=0 and self.hue_value<=20) or (self.hue_value>=160 and self.hue_value<=179)):
            print("Red Brick stack Detected")
            return 1   #1 for the Red Brick stack

        elif(self.hue_value >= 95 and self.hue_value<= 140):
            print("Blue Brick stack Detected")
            return 2   #2 for the Blue Brick stack

        elif((self.hue_value >= 60 and self.hue_value <= 90):
            print("Green Brick stack Detected")
            return 3   #3 for the Green Brick stack
