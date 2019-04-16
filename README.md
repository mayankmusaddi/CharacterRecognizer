CHARACTER RECOGNISER
=====

Developed by @mayankmusaddi with the help of EMNIST by @coopss

##### Description

The Project is a simple Character recogniser which could be trained with any custom dataset in the .mat format. The image file that is to be recognised is passed through several filters which include :
  * Grayscale and Inversion
  * Centering of the character
  * Increasing Contrast by Equalizing Histogram
  * Broadcasting it to a square
  * An OTSU threshold filter

> This project was intended to explore the properties of convolution neural networks (CNN) and see how they compare to recurrent convolution neural networks (RCNN). This was inspired by a [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf "Recurrent Convolutional Neural Network for Object Recognition") I read that details the effectiveness of RCNNs in object recognition as they perform or even out perform their CNN counterparts with fewer parameters. Aside from exploring CNN/RCNN effectiveness, I built a simple interface to test the more challenging [EMNIST dataset](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") dataset (as opposed to the [MNIST dataset](http://yann.lecun.com/exdb/mnist/ "THE MNIST DATABASE of handwritten digits"))

##### Current Implementation
  * Multistack CNN
  * Read in .mat file
  * Currently training on the [byclass dataset](https://cloudstor.aarnet.edu.au/plus/index.php/s/7YXcasTXp727EqB/download) (*direct download link*)
    * See [paper](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters") for more info
  * OTSU Threshold Filter

## Environment

#### Anaconda: Python 3.5.3
  * Tensorflow or tensorflow-gpu (See [here](https://www.tensorflow.org/install/ "Installing TensorFlow") for more info)
  * Keras
  * Numpy
  * Scipy
  * OpenCV 

  Note: All dependencies for current build can be found in dependencies.txt

## Usage
#### [training.py](https://github.com/mayankmusaddi/CharacterRecognizer/training.py)
A training program for classifying the EMNIST dataset

Usage:
    python3 training.py [-h] --file [--width WIDTH] [--height HEIGHT] [--max MAX] [--epochs EPOCHS] [--verbose]

##### Required Arguments:

    -f FILE, --file FILE  Path .mat file data

##### Optional Arguments

    -h, --help            show this help message and exit
    --width WIDTH         Width of the images
    --height HEIGHT       Height of the images
    --max MAX             Max amount of data to use
    --epochs EPOCHS       Number of epochs to train on
    --verbose         Enables verbose printing

#### [recognizer.py](https://github.com/mayankmusaddi/CharacterRecognizer/recognizer.py)
A python app for testing models generated from [training.py](https://github.com/mayankmusaddi/CharacterRecognizer/training.py) on the EMNIST dataset, for labelled images provided by the user.
For the images currently in the [Labelled](https://github.com/mayankmusaddi/CharacterRecognizer/Labelled) folder, the accuracy of the algorithm comes out to be more than 60%.

Usage:
    python3 recognizer.py

##### Directory Briefing:
 - The [Labelled](https://github.com/mayankmusaddi/CharacterRecognizer/Labelled) folder contains all the images for which the first word of its name is the the label for that image.
 - After running the recognizer all the images which are incorrectly predicted are stored in the [ERROR](https://github.com/mayankmusaddi/CharacterRecognizer/ERROR) folder.
 - The images in the [ERROR](https://github.com/mayankmusaddi/CharacterRecognizer/ERROR) folder have their last word in their name as the predicted character.