import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from scipy.misc import imsave, imread, imresize
from matplotlib import pyplot as plt
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle

from improveImage import process

IMAGE_DIRECTORY = './images'
OUTPUT_DIRECTORY = './output'

def load_model(bin_dir):
    ''' Load model from .yaml and the weights from .h5

        Arguments:
            bin_dir: The directory of the bin (normally bin/)

        Returns:
            Loaded model from file
    '''

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def predict(source,model,mapping):
    ''' Called when user presses the predict button.
        Processes the canvas and handles the image.
        Passes the loaded image into the neural network and it makes
        class prediction.
    '''

    # Local functions
    def crop(x):
        # Experimental
        _len = len(x) - 1
        for index, row in enumerate(x[::-1]):
            z_flag = False
            for item in row:
                if item != 0:
                    z_flag = True
                    break
            if z_flag == False:
                x = np.delete(x, _len - index, 0)
        return x

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread(IMAGE_DIRECTORY+"/"+source, mode='L')
    x = np.invert(x)

    ### Experimental
    # # Crop  on rows
    # x = crop(x)
    # x = x.T
    # # Crop on columns
    # x = crop(x)
    # x = x.T

    # Visualize new array
    x = imresize(x,(28,28))

    # 3 kinds of threshold
    ret1,th1 = cv2.threshold(x,127,255,cv2.THRESH_BINARY)
    ret2,th2 = cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(x,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th = th2 #using the 2nd threshold


    # reshape image data for use in neural network
    x = th.reshape(1,28,28,1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Generate response
    prediction = chr(mapping[(int(np.argmax(out, axis=1)[0]))])
    response = {'prediction': prediction,
                'confidence': str(max(out[0]) * 100)[:6]}

    imsave(OUTPUT_DIRECTORY+'/'+prediction+" "+source, th)
    print(response)

def main():
    model = load_model('bin')
    mapping = pickle.load(open('%s/mapping.p' % 'bin', 'rb'))
    for filename in os.listdir(IMAGE_DIRECTORY):
        predict(filename,model,mapping)

main()
