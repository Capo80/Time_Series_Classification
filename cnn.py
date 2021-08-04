import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from utilities import *
from sklearn.model_selection import KFold
from importlib import reload
import classifiers

# GLOBAL variable
x_tr = None
x_ts = None
y_tr = None
y_ts = None
n_classes = 8
input_shape = None
model = None
dataReady = False
trained  = False

def setUp():
    global x_tr, y_tr, x_ts, y_ts, input_shape, dataReady
    # retreiving test and training set
    (x_tr, y_tr) = getTrainingSet()
    (x_ts, y_ts) = getTestSet()

    # scaling data between 0 and 1

    # one hot encoding
    y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
    y_ts = tf.keras.utils.to_categorical(y_ts, num_classes=n_classes)
    input_shape = (x_tr.shape[1],x_tr.shape[2])
    dataReady = True



def startTraining():
    global model, trained

    if(not dataReady):
        print("Setup dataset first !")
        return

    # K-Forld Cross Validation
    n_split=3
    batch_size=256
    for train_index,test_index in KFold(n_split).split(x_tr):
        x_train_fold,x_val_fold=x_tr[train_index],x_tr[test_index]
        y_train_fold,y_val_fold=y_tr[train_index],y_tr[test_index]

        #model = classifiers.get_cnn_standard(input_shape, n_classes)
        #model = classifiers.rest_net(input_shape, n_classes)
        model = classifiers.simple_mlp(input_shape, n_classes)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
        history=model.fit(x_train_fold, y_train_fold, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold), epochs=200, callbacks = [callback])

        print('Model evaluation ', model.evaluate(x_val_fold,y_val_fold))

    # summarize history for loss (best result)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    trained = True

def evaluateOnTestSet():

    if(not trained):
        print("Train the model first !")
        return

    # Testing erformance on test set
    performance = model.evaluate(x_ts, y_ts, verbose=0)
    print("Test performance")
    print(performance)


if __name__ == "__main__":

    while True:
        print("1) Setup dataset")
        print("2) Start Training")
        print("3) Evaluate model on test set")
        print("4) Exit")

        choice = int(input("Your choice: "))

        if(choice == 1):
            setUp()
        elif(choice == 2):
            try:
                # reloading classifier in case of fast modifications
                reload(classifiers)
                startTraining()
            except Exception as e:
                print(str(e))
                print("Error during training")
        elif(choice == 3):
            evaluateOnTestSet()
        elif(choice == 4):
            break
        else:
            print("What ?")
