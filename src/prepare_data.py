from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, pathlib

AUTOTUNE = ""
CLASS_NAMES = ""

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = ""

def prepare_data():

    #TODO: properly prepare color data for dataset training

    #gather images and colors from training set to form pairs
    labeled_ds = load_images('../data/training')

    #prepare image-color pairs for training
    train_ds = prepare_for_training(labeled_ds)

    #create batches for datasets
    image_batch, label_batch = next(iter(train_ds))

    #return the dataset for the estimator
    #TODO: Auto-format data like this:
    #features = {'image 106000 rgb:' : np.array([100, 200, 100]),
    #            'image 106001 rgb:' : np.array([120, 135, 120]),
    #            etc.
    #labels = np.array([106000..107000])
    #return features, labels

    return image_batch, label_batch

def load_images(data_dir):
    """
    Loads images from passed in dir to convert into useful input
    """
    
    # image directory contains directories, each of which are 'classes'
    data_dir = pathlib.Path(data_dir)
    image_cnt = len(list(data_dir.glob('*/*.jpg')))
    print("Image count: ", image_cnt)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    STEPS_PER_EPOCH = np.ceil(image_cnt/BATCH_SIZE)

    # Create a tf dataset for the images and outputs
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    list_cols = tf.data.Dataset.list_files(str(data_dir/'*/*.txt'))    

    for f in list_ds.take(5):
        print(f.numpy())
    for f in list_cols.take(5):
        print(f.numpy())

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(1):
        print("image shape: ", image.numpy().shape)
        print("label: ", label.numpy())

    return labeled_ds
#

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES
#

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
#

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label
#

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    print("preparing images for training...")

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
#
