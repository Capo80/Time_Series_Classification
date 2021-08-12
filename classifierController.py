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
model = None

average_kfold = 0
best_model = None
best_history = None
random.seed("ziofester")

def startTraining():
    global model, x_ts, y_ts, best_model, average_kfold, best_history

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

    # use part on test for validation
    utv = False

    # KFold for model performances evaluation with best model
    n_split = 5
    batch_size = 100
    best_evaluation = [0, 0]
    average_kfold = 0
    for train_index,test_index in KFold(n_split).split(x_tr):
        x_train_fold,x_val_fold=x_tr[train_index],x_tr[test_index]
        y_train_fold,y_val_fold=y_tr[train_index],y_tr[test_index]

        # good ones
        model = classifiers.simple_mlp(input_shape, n_classes)
        #model = classifiers.simple_dnn(input_shape, n_classes)
        #model = classifiers.super_simple_mlp(input_shape, n_classes)

        # sucking models
        #model = classifiers.hybrid_restnet(input_shape, n_classes)
        #model = classifiers.shallow_cnn(input_shape, n_classes)
        #model = classifiers.get_cnn_standard(input_shape, n_classes)
        #model = classifiers.rest_net(input_shape, n_classes)
        if(utv):
            x_val_fold = x_ts[0:int(0.5*x_ts.shape[0])]
            y_val_fold = y_ts[0:int(0.5*y_ts.shape[0])]

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10)
        history = model.fit(x_train_fold, y_train_fold, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold), epochs=200, callbacks = [callback])

        evaluation = model.evaluate(x_val_fold,y_val_fold)
        print(evaluation, best_evaluation)
        average_kfold += evaluation[1]
        print('Current model evaluation ', evaluation)
        if (best_evaluation[1] < evaluation[1]):
            print("New best model found!!")
            best_model = model
            best_history = history
        
    average_kfold = average_kfold / n_split
    if(utv):
        x_ts = x_ts[int(0.5*x_ts.shape[0]):]
        y_ts = y_ts[int(0.5*y_ts.shape[0]):]

    # summarize history for loss (best result)
    plt.plot(best_history.history['loss'])
    plt.plot(best_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def evaluateOnTestSet():

    # Average during Kfold (expected to be better)
    print("Average during Kfold: ", average_kfold)


    # Testing erformance on test set
    performance = best_model.evaluate(x_ts, y_ts, verbose=0)
    print("Test performance on validation set: ", performance)
    


