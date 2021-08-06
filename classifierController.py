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

random.seed(123456789)

def startTraining():
    global model

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
    n_split = 5
    batch_size = 500    # True 0.1 -> 0.9486
                        # True 0.1 + batch = 100 -> 0.957
                        # False -> 0.949                            <- OK
                        # False + Batch=100 ->  0.954
                        # True 0.01 -> 0.948
                        # BrustMode 0.1 -> 0.945
                        # BrustMode 0.01 -> 0.92
                        # Tune
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


def evaluateOnTestSet():

    # Testing erformance on test set
    performance = model.evaluate(x_ts, y_ts, verbose=0)
    print("Test performance")
    print(performance)
