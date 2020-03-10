from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
import pandas as pd
import numpy as np
import matplotlib as plt
import os, sys

from prepare_data import *


NUMERIC_COLUMNS = ['red', 'green', 'blue']

def estimate_palettes():
    print("estimating colors")
    
    feature_columns = []

    #generate the feature_columns
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    #get dataset from prepare_data as a tf.data.Dataset
    dataset = prepare_data()

    train_input_fn

    #set up the linear estimator for the dataset
    linear_est = tf.estimator.LinearClassifier(feature_columns=featurecolumns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    print(result)
#

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True,

if __name__ == "__main__":
    estimate_palettes()
