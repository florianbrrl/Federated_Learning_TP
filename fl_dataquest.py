import numpy as np
import random
import cv2
import os
from imutils import paths # https://github.com/PyImageSearch/imutils

import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf

#==============================================
def load_and_preprocess(paths, verbose=None):
    '''
    Load and preprocess image of the base
    
    Keyword arguments:
    paths --  path of repository of images =>  expects images for each class in seperate dir, e.g all digits in 0
    class in the directory named 0
    verbose -- whant to see a trace during loading ?

    Returns :
    data -- a list of images greyscaled/normalized (0,1)/flattened
    pixels are floating numbers
    labels -- a list of labels/string representing the digit associated
    '''
    data = list() # a list of images
    labels = list()# a list of labels

    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # image/pixel load  from file 
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        # image processings
        image = np.array(im_gray)
        image = image/255 # scale the image to [0, 1] 
        #image=np.expand_dims(image, axis=0) # will move dimension  to (1,28,28)
        #image = image.flatten() # image is now a vector of 28*28 = 784
        data.append(image) # and append to list

        # and extract the class labels => label deduced from file path (a string) !
        label = imgpath.split(os.path.sep)[-2]
        labels.append(label)

        # show an update every `verbose` images
        if verbose is not None and verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))

    # return a tuple of the data and labels
    return data, labels

#==============================================
def get_data(img_path = 'C:/Users/barra/Documents/M2/Menez/TP4/Mnist/Mnist/trainingSet/trainingSet/', verbose = 0) :
    """
    img path to your mnist data folder :  => 42000 fichiers dans 10 répertoires : 0, 1, 2, ...
    """
    if verbose !=0 :
        print(f"\nNous allons travailler avec la base située là : {img_path}")

    # imutils est un module qui offre quelques functions to make basic
    # image processing functions such as translation, rotation,  resizing, skeletonization, ...
    #The paths sub-module of imutils includes a function to recursively find images based on a root directory.
    image_paths = list(paths.list_images(img_path))

    if verbose !=0 :
        print("\n5 premiers fichiers :\n") # Les 5 premiers noms de fichiers 
        print("\n".join(image_paths[:5],))
    
    #fn_list est la liste des noms terminaux (sans le path) de tous les fichiers contenus dans notre base d'images
    fn_list =  [os.path.basename(i) for i in image_paths]
     
    # Pour chacun de ces fichiers, on veut obtenir l'image (liste des pixels) et le label (le chiffre) auquel elle correspond
    # A la fin on obtient deux listes :
    # il : la liste des images/pixels et ll la liste des labels
    if verbose !=0 :
        print("\nMNIST base loading and preprocessing ...\n")

    il, ll =load_and_preprocess(image_paths, verbose=10000)

    if verbose !=0 :
        print("\nLa premiere image : {} \n\t label  : {} \n\t pixels : \n {}".format(fn_list[:1], ll[:1], il[:1]))
        print("Image Shape => ", il[0].shape)
    
    # On a besoin d'une représentation "hot_one" ... des labels voir le doc
    # Comme ceux sont des chiffres .. on peut "binariser"  avec une méthode de sklearn.preprocessing
    lb = skl.preprocessing.LabelBinarizer()     #binarize the labels
    ll = lb.fit_transform(ll)

    if verbose !=0 :
        print("Les 5 premiers labels sous une forme \"hot one\"  :\n", ll[:5])

    # from sklearn => split data into training and test set
    X_train, X_test, y_train, y_test =  train_test_split(il, ll, 
                                                         test_size=0.1, 
                                                         random_state=19)

    # pour voir et etre certain des dimensions (cas MNIST) !
    if verbose !=0 :
        print("Size X_train ", len(X_train), " of " , X_train[0].shape)
        print("Size y_train ", len(y_train), " of " , y_train[0].shape)
        for i in X_train :
            if i.shape != (28,28):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for i in y_train :
            if i.shape != (10,):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if verbose !=0 :            
        print("Size X_test ", len(X_test), " of " , X_test[0].shape)
        print("Size y_test ", len(y_test), " of " , y_test[0].shape)
        for i in X_test :
            if i.shape != (28,28):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for i in y_test :
            if i.shape != (10,):
                print("Alert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return  X_train, X_test, y_train, y_test, X_train[0].shape

#==============================================
def get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose = 0):
    """
    Convertir les images/labels en dataset
    => pipeline Tensorflow  donc passer en "dataset" + batch 

    https://www.tensorflow.org/datasets/performances?hl=fr
    https://www.tensorflow.org/guide/data?hl=fr
    
    Arguments :
         batch_size => attention impact sur les dimensions/shapes des datasets
    """

    dtt = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dtt = dtt.shuffle(buffer_size = len(y_train)) # On mélange tous les éléments
    dtt = dtt.batch(batch_size)
    if verbose !=0 :
        print("\nDataset for train :\n", dtt.element_spec)

    dts = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dts = dts.batch(batch_size)
    if verbose !=0 :
        print("\nDataset for test :\n", dts.element_spec ,"\n")

        
    return dtt, dts
