import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from libraries.utilities import *
from libraries.constants import * 
from sklearn.model_selection import KFold, GridSearchCV
import random, os, string
from importlib import reload
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import time, datetime
import libraries.parameters
from itertools import combinations

# GLOBAL variables
model = None

ensamble_models = []
average_kfold = 0
best_model = None
best_history = None
best_evaluation = 0
training_time = 0
last_model_name = ""
random.seed("ziofester")
best_ensamble = []

def startTraining():
    global model, x_ts, y_ts, best_model, best_evaluation, average_kfold, best_history, training_time, last_model_name
    #reload(parameters)
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
    n_split = parameters.KFOLD_SPLIT
    batch_size = parameters.BATCH_SIZE
    best_evaluation = [1000, 0]
    average_kfold = 0
    training_time = 0
    start_time = time.monotonic()

    #print(parameters.BATCH_SIZE)
    training_function = parameters.FUNC_NAME

    last_model_name = training_function.__name__
    for train_index,test_index in KFold(n_split).split(x_tr):
        #print(train_index, test_index)
        x_train_fold,x_val_fold=x_tr[train_index],x_tr[test_index]
        y_train_fold,y_val_fold=y_tr[train_index],y_tr[test_index]

        if(utv):
            x_val_fold = x_ts[0:int(0.5*x_ts.shape[0])]
            y_val_fold = y_ts[0:int(0.5*y_ts.shape[0])]

        model = training_function(input_shape, n_classes)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=parameters.PATIENCE)

        history = model.fit(x_train_fold, y_train_fold, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold), epochs=parameters.EPOCH, callbacks = [callback], verbose=0)

        evaluation = model.evaluate(x_val_fold,y_val_fold, verbose=0)
        #print(evaluation, best_evaluation)
        average_kfold += evaluation[1]
        print('Current model evaluation ', evaluation)
        if (best_evaluation[0] > evaluation[0]):
            #print("New best model found!!")
            best_model = model
            best_history = history
            best_evaluation = evaluation

    average_kfold = average_kfold / n_split
    training_time = time.monotonic() - start_time
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
    #plt.show()

def ensambleStartTraining():
    global ensamble_models, x_ts, y_ts, best_model, best_evaluation, average_kfold, best_history, training_time, last_model_name
    reload(parameters)

    # KFold for model performances evaluation with best model
    n_split = parameters.KFOLD_SPLIT
    batch_size = parameters.BATCH_SIZE
    best_evaluation = [1000, 0]
    average_kfold = 0
    training_time = 0
    start_time = time.monotonic()

    #print(parameters.BATCH_SIZE)
    training_function = parameters.FUNC_NAME

    last_model_name = training_function.__name__
    n_models = 5
    for ind in range(0, n_models):
        ens_size = int(x_tr.shape[0]/n_classes)
        ens_dataset_x = x_tr[ind * ens_size: (ind+1) * ens_size]
        ens_dataset_y = y_tr[ind * ens_size: (ind+1) * ens_size]
        
        val_size = int(ens_dataset_x.shape[0]*0.2)
        ens_train_x = ens_dataset_x[val_size:]
        ens_train_y = ens_dataset_y[val_size:]
        ens_val_x = ens_dataset_x[:val_size]
        ens_val_y = ens_dataset_y[:val_size]
        
        model = training_function(input_shape, n_classes)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=parameters.PATIENCE)

        history = model.fit(ens_train_x, ens_train_y, batch_size=batch_size, validation_data=(ens_val_x, ens_val_y), epochs=parameters.EPOCH, callbacks = [callback], verbose=1)

        evaluation = model.evaluate(ens_val_x,ens_val_y, verbose=0)

        ensamble_models.append(model)

        print("Current evaluation: ", evaluation)

    training_time = time.monotonic() - start_time

    # summarize history for loss (best result)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()

def ensambleEvaluate():

    global best_ensamble
    best_evaluation = 0

    for model in ensamble_models:
        print("Single model prediction: ", model.evaluate(x_ts, y_ts, verbose=0))

    for i in range(2, len(ensamble_models)+1):
        for comb in combinations(ensamble_models, i):
            yhats = [model.predict(x_ts) for model in comb]
            yhats = np.array(yhats)
            # sum across ensemble members
            summed = np.sum(yhats, axis=0)
            # argmax across classes
            result = np.argmax(summed, axis=1)

            result = tf.keras.utils.to_categorical(result, num_classes=n_classes)

            curr_evaluation = accuracy_score(y_ts, result)
            print("Ensamble accuracy: ",curr_evaluation )

            if curr_evaluation > best_evaluation:
                best_evaluation = curr_evaluation
                best_ensamble = comb
    print("Best evaluation: ", best_evaluation)

    #save results on file
    write_line_to_csv("results.csv", last_model_name, datetime.datetime.now(), training_time, "ensamble", best_evaluation, parameters.AUGMENT, parameters.EPOCH, parameters.BATCH_SIZE, parameters.SEED, parameters.KFOLD_SPLIT, parameters.PATIENCE)


def evaluateOnTestSet():

    #training time
    print("Training time: ", training_time)

    # Average during Kfold (expected to be better)
    print("Average during Kfold: ", average_kfold)

    # Best model performance
    print("Best model performance: ", best_evaluation)

    # Testing erformance on test set
    performance = best_model.evaluate(x_ts, y_ts, verbose=0)
    print("Test performance on test set: ", performance)

    #save results on file
    write_line_to_csv("results.csv", last_model_name, datetime.datetime.now(), training_time, average_kfold, performance[1], parameters.AUGMENT, parameters.EPOCH, parameters.BATCH_SIZE, parameters.SEED, parameters.KFOLD_SPLIT, parameters.PATIENCE)

    return (performance, best_model)

def saveLastModel():
    folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    os.mkdir(models_path + "/" + folder)
    best_model.save(models_path + "/" + folder)

def saveEnsamble():
    folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    os.mkdir(models_path + "/" + folder)
    for i in range(0, len(best_ensamble)):
        os.mkdir(models_path + "/" + folder + "/model" + str(i))
        ensamble_models[i].save(models_path + "/" + folder + "/model" + str(i))