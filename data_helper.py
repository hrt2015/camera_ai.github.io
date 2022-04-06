import cv2
import numpy as np
import keras
import cv2
import os
import matplotlib.pyplot as plt
def readfile_to_dict(filename):
    'Read text file and return it as dictionary'
    d = {}
    f = open(filename)
    for line in f:
        # print(str(line))
        if line != '\n':
            (key, val) = line.split()
            d[key] = int(val)

    return d
readfile_to_dict("E:\MINGTOM\\NEW\\DATASET\\a01")
