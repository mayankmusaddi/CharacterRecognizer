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
    show(img,'original')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 8:
                img[i][j]=0
    show(img,'edited')
    
    gray = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8)) # Perform noise filtering
    show(img,'noiseremoved')
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    print("Bounding Box",x,y,w,h)
    ## Error for 91 6
    img = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    show(img,'cropped')

    img = cv2.equalizeHist(img)
    show(img,'enhanced')

    img = broadcast(img)
    show(img,'Broadcasted')

    img = imresize(img,(28,28))
    show(img,'Resized')
    return img

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

imag = imread(IMAGE_DIRECTORY+"/9 130 8.png")
process(imag)
