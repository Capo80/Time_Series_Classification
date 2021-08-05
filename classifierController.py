import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from utilities import *
from sklearn.model_selection import KFold, GridSearchCV
import classifiers
import random
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# GLOBAL variables
x_tr = None
x_ts = None
y_tr = None
y_ts = None
n_classes = 8
input_shape = None
model = None
dataReady = False
trained  = False

random.seed(123456789)

def setUp(dataAugumentationRatio=0, infraTimeAcc=False, infraPerc=0.3):
    global x_tr, y_tr, x_ts, y_ts, input_shape, dataReady

    # retreiving test and training set
    print("Loading Training set")
    (x_tr, y_tr) = getTrainingSet()
    print("Loading Test Set")
    (x_ts, y_ts) = getTestSet()

    print("Adjusting data")
    # adjusting data, using only 6 digits after 0
    for i in range(0, x_tr.shape[0]):
        for j in range(0, x_tr.shape[1]):
            for k in range(0, x_tr.shape[2]):
                # if using relu, negative input must be handled
                x_tr[i][j][k] = float("%.6f"%x_tr[i][j][k])

    for i in range(0, x_ts.shape[0]):
        for j in range(0, x_ts.shape[1]):
            for k in range(0, x_ts.shape[2]):
                # if using relu, negative input must be handled
                x_ts[i][j][k] = float("%.6f"%x_ts[i][j][k])

    # data augumentation
    if(dataAugumentationRatio != 0):

        augShape = (int(dataAugumentationRatio*x_tr.shape[0]), x_tr.shape[1], x_tr.shape[2])
        print("Adding %d Training Set Entries" % augShape[0])

        # creating augumented np array
        train = np.empty(augShape)
        train_l = np.empty(augShape[0])

        # populating arrays
        for i in range(0, augShape[0]):
            r = random.random()
            if (infraTimeAcc and r <= infraPerc):
                start = random.randint(10, augShape[1]-10)
                end = random.randint(start, augShape[1])
                interval = int(float(end-start)/2)
            else:
                start = 0
                end = augShape[1]

            posR = random.random()
            negR = -posR
            pintR = random.randint(0,3)
            nintR = -pintR
            for j in range(start, end):
                for k in range(0, augShape[2]):
                    ii = i % (x_tr.shape[0])
                    if(i%2==0):
                        # adding an 'accellerated (existing) motion'
                        if (infraTimeAcc and r <= infraPerc):
                            if(j <= start+interval):
                                train[i][j][k] = x_tr[ii][j][k]+(posR+pintR)*int((j)/float(start+interval))
                            else:
                                train[i][j][k] = x_tr[ii][j][k]-(posR+pintR)*int((j)/float(start+interval))
                        else:
                            train[i][j][k] = x_tr[ii][j][k]+(posR+pintR)
                    else:
                        # adding a 'decellerated (existing) motion'
                        if (infraTimeAcc and r <= infraPerc):
                            if(j <= start+interval):
                                train[i][j][k] = x_tr[ii][j][k]+(negR+nintR)*int((j)/float(start+interval))
                            else:
                                train[i][j][k] = x_tr[ii][j][k]-(negR+nintR)*int((j)/float(start+interval))
                        else:
                            train[i][j][k] = x_tr[ii][j][k]+(negR+nintR)
                    train_l[i] = y_tr[ii]

        # merging arrays
        x_tr = np.append(x_tr, train, axis=0)
        y_tr = np.append(y_tr, train_l, axis=0)

        print("Data augumented, new shape: ", x_tr.shape)



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

    # TODO automatic parameter tuning
    # K-Forld Cross Validation with GridSearchCV for Automatic Tuning (not working...)
    """
    n_split=5
    batch_size=128
    # define the grid search parameters
    batch_size = [10, 60]
    epochs = [10]

    batch_size = [10, 60, 128, 256, 512, 1024, 2048]
    epochs = [10, 50, 100]
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    #param_grid = dict(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    model = KerasClassifier(build_fn= classifiers.super_simple_mlp, input_shape=input_shape, n_classes=n_classes, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy",
                    n_jobs=-1, refit=True, cv=n_split, return_train_score=True)
    # return_train_score to identify over/under - fitting
    grid.fit(x_tr, y_tr)
    print(grid_result.best_params_)

    # TODO: continue here
    # re-training on all dataset and with best results
    val_perc = 0.2
    x_train =
    y_train =
    x_val =
    y_val =

    best_batch_size = grid_result.best_params_["batch_size"]
    best_epochs = grid_result.best_params_["ephocs"]
    best_lr = grid_result.best_params_["learn_rate"]
    best_mom = grid_result.best_params_["momentum"]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
    history=model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold), epochs=200, callbacks = [callback])

    print('Model evaluation ', model.evaluate(x_val_fold,y_val_fold))
    """

    # KFold for model performances evaluation with best model
    n_split = 10
    n_split=5
    batch_size=512
    for train_index,test_index in KFold(n_split).split(x_tr):
        x_train_fold,x_val_fold=x_tr[train_index],x_tr[test_index]
        y_train_fold,y_val_fold=y_tr[train_index],y_tr[test_index]

        #model = classifiers.get_cnn_standard(input_shape, n_classes)
        #model = classifiers.rest_net(input_shape, n_classes)
        model = classifiers.simple_mlp(input_shape, n_classes)
        #model = classifiers.simple_dnn(input_shape, n_classes)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
        history=model.fit(x_train_fold, y_train_fold, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold), epochs=200, callbacks = [callback])

        print('Model evaluation ', model.evaluate(x_val_fold,y_val_fold))
        break

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
