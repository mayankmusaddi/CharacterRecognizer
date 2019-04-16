import os
import cv2
import numpy as np
from scipy.misc import imsave, imread, imresize
from matplotlib import pyplot as plt

IMAGE_DIRECTORY = './images'
OUTPUT_DIRECTORY = './output'

def process(img):
    # read parsed image back in 8-bit, black and white mode (L)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.invert(img)
    # show(img,'original')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 8:
                img[i][j]=0
    # show(img,'edited')

    img = img[ int(img.shape[0]/3): int(2*img.shape[0]/3), int(img.shape[1]/3): int(2*img.shape[1]/3)]
    # show(img,'centre')

    sd = np.std(img)
    # print("STANDARD DEV :: ",sd)
    return sd

def broadcast(img):
    height, width = img.shape
    x = height+2 if height > width else width+2
    y = height+2 if height > width else width+2
    square= np.zeros((x,y), np.uint8)
    square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
    return square

def show(img,title):
    print(title)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            print(img[i,j],end='|')
        print()

    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def main():
    path = '../Training Images/Labelled'
    for fln in os.listdir(path):
        file_name = os.path.join(path, fln)
        img = imread(file_name)
        sd = process(img)
        if(sd <= 10):
            # os.rename(file_name,'../Training Images/Blanks/Letters/'+fln)
            print("+++++++++++++++++++++++++++",fln," ==> ",sd)
        else:
            print(fln," ==> ",sd)

# imag = imread(IMAGE_DIRECTORY+"/85 3.png")
# process(imag)
main()
