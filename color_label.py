#Script for determining colour within the bounding box
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

class colorlabel:
    def __init__(self):
        self.hue_value = None
        self.hue_image = None


    def label(self,image):
        self.hue_image = image[:,:,1] #Hue channel
        hist = cv2.calcHist([image],[0],None,[256],[0,256])
        self.hue_value = np.where(hist == np.amax(hist))
        self.hue_value = self.hue_value[0][0]
        print("self.hue_value:",self.hue_value)

        if((self.hue_value>=0 and self.hue_value<=40) or (self.hue_value>=160 and self.hue_value<=179)):
            print("Red Brick Detected")
            return 1   #1 for the Red Brick pile

        elif(self.hue_value >= 90 and self.hue_value<= 140):
            print("Blue Brick Detected")
            return 2   #2 for the Blue Brick pile
