import numpy
import numpy.core.multiarray
import cv2
from time import time
# from numpy.core.multiarray import *
from . import libmain
t1 = time()
txt = libmain.TextDetection(cv2.imread("flowers.jpeg"))
txt.Run_Filters()
t2 = time()
print("Time: ", t2-t1)