from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, pathlib

def load_images(data_dir):
    image_cnt = len(list(data_dir.glob('*/*.jpg')))
    print(image_cnt)
#

def extract_colors():
    
#

if __name__ == '__main__':
    load_images('../img/palettes'):
