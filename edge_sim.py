import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

import fl_dataquest
import fl_model

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def make_edges(image_list, label_list, num_edges=10, verbose = 0):
    '''    
    Shard the sequence among the edges that are going to federate learning
    
    Arguments:
            image_list --  a list of numpy arrays of training images
            label_list -- a list of binarized labels for each image
    
            num_edge --  number of federated members (edges)
    
    return:
           a dictionary with keys that are edge names
                                and value are data shards - tuple of two lists : images list and labels list.
    '''

    #create a list of names of edge => "edge_0", "edge_1", ...
    edge_names = ['{}_{}'.format("edge", i) for i in range(num_edges)]
    if verbose != 0 :
        print(edge_names)

    # zip => On fabrique un zip-iterator qui contient l'agregation de deux listes : celle des images et celle des labels
    data = list(zip(image_list, label_list)) # we make a list from this zip-iterator 
    random.shuffle(data)                           # randomize the list .. to not allocate the same items to same edge

    #data = [(i1,l1), (i2,l2), etc] une liste de tuples car (i1,l1) est un tuple
    
    #shard data and place at each edge => on fait ca par slicing 
    size = len(data)//num_edges # number of items for each edge
    shard = [data[i:i + size] for i in range(0, size*num_edges, size)]  # shard is a list of tuples
    # edges dictonnary => Chaque edge i "recoit" son shard
    edges = {edge_names[i] : shard[i] for i in range(num_edges)}

    if verbose != 0 :
        # par exemple le edge_0
        tmp = edges["edge"+"_0"] # tmp is the list of tuples allocated to the edge_0
        print("First tuple of the edge 0 ", tmp[0]) # 

    #A partir du dictonnaire "edges", on fait un nouveau dico dans
    #lequels les listes (images et labels) s'intègrent dans un dataset.
    #On en profite pour mélanger et batcher pour chaque edge

    edges_batched = dict() 
    for (edge_name, st) in edges.items():
        # st est liste de tuples (chaque tuple contient une liste d'images et une liste de labels)
        il, ll = zip(*st)   # unzip st => separe les listes aggrégées dans les tuples et rend deux tuples
        il = list(il) # make a list from the tuple => il is the list of images of current edge
        ll = list(ll) # ll : labels associated to images of current edge

        print("edge name : {} | #images : {} | #labels : {} | {} | {}".format(edge_name, len(il), len(ll), il[0].shape, ll[0].shape ))

        # Make a dataset pour chaque edge
        dataset = tf.data.Dataset.from_tensor_slices((il, ll)) # un dataset avec deux elements (/deux listes)
        buffer_size = len(ll) # le dataset contient autant d'items que le nombre de labels ou d'images
        dataset = dataset.shuffle(buffer_size) # l'ordre des images dans le edge n'est plus "relevant" 
        batch_size=32 
        dataset = dataset.batch(batch_size)
        
        edges_batched[edge_name] =  dataset

    return edges_batched 

#===========================================

def weight_scaling_factor(edges, edge_name, verbose = 0):
    """
    Compute the scaling factor fort his edge : local_count/global_count
    """
    
    alledge_names = list(edges.keys()) # liste de tous les noms d'edge

    #Calculer le batch size du edge => bs
    t = edges[edge_name] # t is a dataset  associated to the edge_name key in the dict
    if verbose != 0:
        print(f"\tBatched Dataset associated to {edge_name} : \n{t.element_spec}\n")

    l = list(t) # make a list from this dataset pour pouvoir indexer
    if verbose != 0:
        print(f"\tNombre de batch pour ce edge : {len(l)}")

    ft = l[0] # first element of the dataset
    if verbose != 0:
        print(f"\tfirst : {ft}") # on obtient deux tenseurs le premier contient 32 images (28x28) et le second 32 vecteurs hot_one

    li = ft[0] # on recupere le premier des deux tenseurs
    bs = li.shape[0] # et on recupere le batch size
    if verbose != 0:
        print(f"\tbatch size of this edge : {bs}")

    # first calculate the total training data points across edges
    #  tf.data.experimental.cardinality returns a tensor => that is why .numpy()
    global_count = sum([tf.data.experimental.cardinality(edges[edge_name]).numpy() for edge_name in alledge_names])*bs
    if verbose != 0:
        print(f"\tnumber of images over ALL edges : {global_count}")
 
    # then get the total number of data points held by the edge "edge_name"
    local_count = tf.data.experimental.cardinality(edges[edge_name]).numpy()*bs
    if verbose != 0:
        print(f"\tnumber of image in this  edge : {local_count}")
    
    return local_count/global_count

def scale_model_weights(weight, scalar):
    '''
    function for scaling a edge models weights
    '''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
        
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''
    Return the sum of the listed scaled weights.
    The is equivalent to scaled avg of the weights
    '''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # on récupère les données en mémoire 
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data( 'C:/Users/barra/Documents/M2/Menez/TP4/Mnist/Mnist/trainingSet/trainingSet/', verbose=1)  #input_shape = (28, 28)  

    # on en fait un/deux pipelines  BATCHED : dtt et dts
    dtt, dts = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=1)
    
    #=======================================================
    # Le modele central se débrouille seul dans un premier temps
    verbose = 0

    # Instancie le modele
    cmodel = fl_model.MyModel(input_shape, nbclasses=10)
    # L'entraine
    cmodel.fit_it(trains = dtt, epochs=10, tests = dts, verbose=verbose)
    lossbyc, accuracybyc  = cmodel.evaluate(dts, verbose=verbose)  # à partir du score, on recupere la loss et l'accuracy 
    print("==> Loss on tests : {}  & Accuracy :  {}".format(lossbyc, accuracybyc))

    # weights of the central model will be uses as the initial weights for all local models
    central_weights = cmodel.get_weights()  
   
   #=======================================================
    # Puis le modele se fabrique selon FedAVG
    # pour cela on crée des "edges" qui vont fonctionner en FL
    # => chaque edge se voit attribuer un bout de training
    num_edges = 10
    edge_epoch = 1
    edges = make_edges(X_train, y_train, num_edges=num_edges)

    # Les modeles en edge contriburaient au model central
    print("\nEdge models : Start ! --------------------------------\n")
    verbose =0

    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    edges_names= list(edges.keys())
    random.shuffle(edges_names)

    # a round = loop through each edge and create new local model
    for en in edges_names:    
        print(f"Processing : {en}", end=" ")
        dtt =  edges[en] #get dataset of the current edge

        #set local model weight to the weight of the central model
        lmodel = fl_model.MyModel(input_shape, nbclasses=10)
        lmodel.set_weights(central_weights)

        # run a fit of this edge model
        lmodel.fit_it(trains = dtt, epochs=edge_epoch, tests = dts, verbose=verbose)

        # each edge's model must be scaled => first compute the factor 
        scaling_factor = weight_scaling_factor(edges, en) #=> local_count/global_count
        print(f"\n\tScaling Factor :  ({scaling_factor})")
        #scale the model weights 
        scaled_weights = scale_model_weights(lmodel.get_weights(), scaling_factor)
        # and add to list
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()

    #end the round
    
   #=======================================================
    # Update central model : from edges contributions
    average_weights = sum_scaled_weights(scaled_local_weight_list) # to get the average over all the local model,
                                                                                                                 # we simply take the sum of the scaled weights
                                                                                                                 
    cmodel.set_weights(average_weights) #Update central model
    
    #=======================================================
    print("\nEvaluation du modele central avec les poids calcules par les edges (Federated Learning) : \n")
    lossbyfedavg, accuracybyfedavg = cmodel.evaluate(dts, verbose=verbose)  # à partir du score, on recupere la loss et l'accuracy 
    print("FedAvg  ==> Loss on tests : {}  & Accuracy :  {}".format(lossbyfedavg, accuracybyfedavg))

    # Remember without FedAvg
    print("Central ==> Loss on tests : {}  & Accuracy :  {}".format(lossbyc, accuracybyc))
