"""

Ce module crée le modèle qui va être utilisé par tous les particpants du FedAvg.
on rappele que tous le monde doit utilisé la mêm structure de modèle.


"""
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

import fl_dataquest

#=======================================================
class MyModel():
    """
    On définit le model qui sera utilise pour valider le FedAVG
    """
    
    def __init__(self, input_shape, nbclasses):
        """
        On construit un MLP NN 
        """
        model = Sequential()
        model.add(Input(input_shape))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(nbclasses))
        model.add(Activation("softmax"))

        # On a donc un model MLPNN constuit sur Sequential
        # Pour classer "nbclasses" d'images  de shape précisée.
        self.model = model
        self.input_shape= input_shape
        self.classes = nbclasses

        self.loss_fn = 'categorical_crossentropy'
        self.model.compile(optimizer="SGD", loss=self.loss_fn, metrics=["accuracy"]) # ou "adam"
    
    def set_weights(self, w):
        self.model.set_weights(w)

    def fit_it(self, trains, epochs, tests, verbose) :
        """
        Entrainement du modele 
        """
        self.trains = trains
        self.tests = tests
        self.history = self.model.fit(trains,
                                      epochs=epochs,
                                      validation_data=tests,    # https://stackoverflow.com/questions/67199384/difference-between-validation-accuracy-and-results-from-model-evaluate
                                      verbose=verbose)

    def summary(self):
        return self.model.summary()

    def evaluate(self, tests, verbose):
        """
        evaluate model on tests
        return  la loss et l'accuracy 
        """
        return self.model.evaluate(tests, verbose=verbose)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def pretty_print_layers(self):
        """ Pretty print weights of the layers of thee model """
        for layer_i in range(len(self.model.layers)):    # loop over each layer and get weights and biases
            l = self.model.layers[layer_i].get_weights()
            print(type(l))
            if len(l) != 0 :
                w = self.model.layers[layer_i].get_weights()[0] # weight 
                b = self.model.layers[layer_i].get_weights()[1] # bias
                print('Layer {} has weights of shape {} and biases of shape {}' .format(layer_i, np.shape(w), np.shape(b)))


#=======================================================
if __name__ == '__main__':
    verbose = 2
        
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data( 'C:/Users/barra/Documents/M2/Menez/TP4/Mnist/Mnist/trainingSet/trainingSet/')
    dtt, dts = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test)

    # Instancie le modele
    m = MyModel(input_shape, nbclasses=10)
    # L'entraine
    m.fit_it(trains = dtt, epochs=10, tests = dts, verbose=verbose)
    
    if verbose != 0:
        m.summary()
    
    # On évalue ce modele  => prediction with an approach based on a model "at the center"
    print("\nEvaluation du modele post-fit : \n")
    loss, accuracy  = m.evaluate(dts, verbose=verbose)  # à partir du score, on recupere la loss et l'accuracy 
    print("==> Loss on tests : {}  & Accuracy :  {}".format(loss, accuracy))

    #=======================================================    
    # Jouez avec les W
    global_weights = m.get_weights()  #1) On recupere les poids du modele
    m.set_weights(global_weights)            
    # et voir si l'evaluation du model persiste
    print("\nEvaluation du modele  (post-set_weights) : \n")
    loss, accuracy  = m.evaluate(dts, verbose=verbose)  # à partir du score, on recupere la loss et l'accuracy 
    print("==> Loss on tests : {}  & Accuracy :  {}".format(loss, accuracy))

    print("\nModel weights pretty print !\n")     
    m. pretty_print_layers()

