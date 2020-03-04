from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, pathlib

def load_images(data_dir):
    """
    Loads images from passed in dir to convert into useful input
    """
    
    # image directory contains directories, each of which are 'classes'
    image_cnt = len(list(data_dir.glob('*/*.jpg')))
    print("Image count: ", image_cnt)

    # Create a tf dataset for the images
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
#

def extract_colors():
   print("Extracting colors (not really)") 
#

if __name__ == '__main__':
    load_images('../img/palettes'):
