import cv2
import numpy as np
import os
import pandas as pd
from skimage.io import imread

img_path = '/data/training/bgs/'
for img in os.listdir('..' + img_path):
    npy_name = img + '.npy'
    npy_path = '..' + '/data/training/npy/' + npy_name
    
    # for convenience, don't re-generate features
    #if os.path.isfile(npy_path):
    #    continue

    path = '..' + img_path + img

    image = cv2.imread(path)
    
    IMG_HEIGHT = image.shape[0]
    IMG_WIDTH = image.shape[1]
    
    image = cv2.resize(image, dsize=(int(IMG_HEIGHT/2),int(IMG_WIDTH/2)), interpolation=cv2.INTER_CUBIC)

    IMG_HEIGHT = image.shape[0]
    IMG_WIDTH = image.shape[1]
    
    print('image', img, ' shape:', image.shape)


    # create 2d feature matrix for image width, height
    feature_matrix = np.zeros((IMG_HEIGHT, IMG_WIDTH))  
    print('matrix shape:', feature_matrix.shape)

    # populate feature matrix with the 3 color channels average values
    for i in range(0, IMG_HEIGHT):
        for j in range(0, IMG_WIDTH):
            
            # if the image is grayscale it won't contain color channel data
            if len(image.shape) == 2:
                feature_matrix[i][j] = ((int(image[i,j])))
            else:
                feature_matrix[i][j] = ((int(image[i,j,0]) +
                                        int(image[i,j,1]) +
                                        int(image[i,j,2])) / 
                                        image.shape[2])

    # reshape to 1d array
    features = np.reshape(feature_matrix, (IMG_HEIGHT*IMG_WIDTH))
    print("features shape:", features.shape)

    print('saving to:', npy_path)

    # save features to npy file
    np.save(npy_path, features)
