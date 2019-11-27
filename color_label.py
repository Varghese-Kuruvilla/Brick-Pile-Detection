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
        ''' Determines the color of the brick stack
        Returns: Color code indicating color of the brick stack
        1: Red brick stack , 2: Blue brick stack, 3: Green Brick stack '''
        self.hue_image = image[:,:,0] #Hue channel
        hist = cv2.calcHist([image],[0],None,[180],[0,180])
        self.hue_value = np.where(hist == np.amax(hist))
        self.hue_value = self.hue_value[0][0]
        print("self.hue_value:",self.hue_value)

        if((self.hue_value>=0 and self.hue_value<=20) or (self.hue_value>=160 and self.hue_value<=179)):
            print("Red Brick Detected")
            return 1   

        elif(self.hue_value >= 99 and self.hue_value<= 130):
            print("Blue Brick Detected")
            return 2
        
        elif(self.hue_value >=30 and self.hue_value<=95):
            print("Green Brick Detected")
            return 3    
