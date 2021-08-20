import numpy as np
from libraries.constants import *
import random
import sklearn
import tensorflow as tf
import csv
import libraries.parameters as parameters

# GLOBAL variables
x_tr = None
x_ts = None
y_tr = None
y_ts = None
n_classes = 8
input_shape = None

def write_line_to_csv(filename, *kargs):
    with open(filename, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        row = [str(i) for i in kargs]
        spamwriter.writerow(row)

def setUp(dataAugumentationRatio=0, infraTimeAcc=False, infraPerc=0.3, random=0, seed=0, approx=1):
    global x_tr, y_tr, x_ts, y_ts, input_shape

    # retreiving test and training set
    if (random):
        print("Loading random Test and Training")
        (x_tr, y_tr, x_ts, y_ts) = getRandomTestTrain(seed=seed)
    else:
        print("Loading Training set")
        (x_tr, y_tr) = getTrainingSet()
        print("Loading Test Set")
        (x_ts, y_ts) = getTestSet()

    # data augumentation
    if(dataAugumentationRatio != 0):
        (x_tr, y_tr) = augumentDataset(x_tr, y_tr, infraTimeAcc, infraPerc, dataAugumentationRatio)


    if (approx):
        # adjusting data, using only 6 digits after 0
        print("Adjusting data")
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


    # one hot encoding
    y_tr = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
    y_ts = tf.keras.utils.to_categorical(y_ts, num_classes=n_classes)
    input_shape = (x_tr.shape[1],x_tr.shape[2])
    dataReady = True

def getRandomTestTrain(percTest=0.3,seed=parameters.SEED):

    random.seed(seed)

    train_x = np.loadtxt(open(train_x_filename, "rb"), delimiter=",")
    train_y = np.loadtxt(open(train_y_filename, "rb"), delimiter=",")
    train_z = np.loadtxt(open(train_z_filename, "rb"), delimiter=",")
    train_l = np.loadtxt(open(train_label_filename, "rb"), delimiter=",")

    axis = (train_x, train_y, train_z)
    samples = int(train_x.shape[0])
    testSamples = int(percTest*train_x.shape[0])
    timeUnits = train_x.shape[1]

    train = np.empty(shape=(samples, timeUnits, 3))
    train_label = np.empty(shape=(samples))

    test = np.empty(shape=(testSamples, timeUnits, 3))
    test_label = np.empty(shape=(testSamples))

    #fill train set
    for sample in range(0, samples):
        for time in range(0, timeUnits):
            for k in range(0, len(axis)):
                train[sample][time][k] = (axis[k])[sample][time]
        train_label[sample] = train_l[sample]

    # shuffle training set
    train, train_label = sklearn.utils.shuffle(train, train_label)

    #randomly remove percTest of train for test
    for i in range(0, testSamples):
        random_index = random.randint(0, samples-1-i)
        test[i] = train[random_index]
        #print(random_index)
        test_label[i] = train_label[random_index]
        train = np.delete(train, random_index, 0)
        train_label = np.delete(train_label, random_index, 0)

    print("Train shape: ", train.shape, "Train label shape: ", train_label.shape)
    print("Test shape: ", test.shape, "Test label shape: ", test_label.shape)

    # USED ONLY TO DEBUG Evaluation Procedure made by the teacher, set to False
    saveTestSet = False
    # writing test set into a file
    if(saveTestSet):
        np.savetxt(test_x_filename, [ [test[i][j][0] for j in range(0,test.shape[1])] for i in range(0, test.shape[0]) ], delimiter=",", fmt="%s")
        np.savetxt(test_y_filename, [ [test[i][j][1] for j in range(0,test.shape[1])] for i in range(0, test.shape[0]) ], delimiter=",", fmt="%s")
        np.savetxt(test_z_filename, [ [test[i][j][2] for j in range(0,test.shape[1])] for i in range(0, test.shape[0]) ], delimiter=",", fmt="%s")
        np.savetxt(test_label_filename, test_label, delimiter=",", fmt="%i")

    return (train, train_label, test, test_label)

def getTrainingSet(perc=0.7):

    #training shape will be like (5000, 315, 3)
    train_x = np.loadtxt(open(train_x_filename, "rb"), delimiter=",")
    train_y = np.loadtxt(open(train_y_filename, "rb"), delimiter=",")
    train_z = np.loadtxt(open(train_z_filename, "rb"), delimiter=",")
    train_l = np.loadtxt(open(train_label_filename, "rb"), delimiter=",")

    axis = (train_x, train_y, train_z)
    samples = int(perc*train_x.shape[0])
    timeUnits = train_x.shape[1]

    train = np.empty(shape=(samples, timeUnits, 3))
    train_label = np.empty(shape=(samples))

    # extracting the first 'perc' of training test
    for sample in range(0, samples):
        for time in range(0, timeUnits):
            for k in range(0, len(axis)):
                train[sample][time][k] = (axis[k])[sample][time]
        train_label[sample] = train_l[sample]

    print("Train shape: ", train.shape, "Train label shape: ", train_label.shape)
    return (train, train_label)


def loadTSFile():
    test_x = np.loadtxt(open(test_x_filename, "rb"), delimiter=",")
    test_y = np.loadtxt(open(test_y_filename, "rb"), delimiter=",")
    test_z = np.loadtxt(open(test_z_filename, "rb"), delimiter=",")
    test_l = np.loadtxt(open(test_label_filename, "rb"), delimiter=",")

    axis = (test_x, test_y, test_z)
    samples = int(test_x.shape[0])
    timeUnits = test_x.shape[1]

    test = np.empty(shape=(samples, timeUnits, 3))
    test_label = np.empty(shape=(samples))

    # populating test set
    for sample in range(0, samples):
        for time in range(0, timeUnits):
            for k in range(0, len(axis)):
                test[test_x.shape[0]-sample-1][time][k] = (axis[k])[sample][time]
        test_label[test_x.shape[0]-sample-1] = test_l[sample]

    test_label = tf.keras.utils.to_categorical(test_label, num_classes=n_classes)
    dataReady = True

    print("Test shape: ", test.shape, "Test label shape: ", test_label.shape)
    return (test, test_label)


def getTestSet(perc=0.3):

    test_x = np.loadtxt(open(train_x_filename, "rb"), delimiter=",")
    test_y = np.loadtxt(open(train_y_filename, "rb"), delimiter=",")
    test_z = np.loadtxt(open(train_z_filename, "rb"), delimiter=",")
    test_l = np.loadtxt(open(train_label_filename, "rb"), delimiter=",")

    axis = (test_x, test_y, test_z)
    samples = int(perc*test_x.shape[0])
    timeUnits = test_x.shape[1]

    test = np.empty(shape=(samples, timeUnits, 3))
    test_label = np.empty(shape=(samples))

    # extracting the LAST 'perc' of training test
    for sample in range(test_x.shape[0]-samples, test_x.shape[0]):
        for time in range(0, timeUnits):
            for k in range(0, len(axis)):
                test[test_x.shape[0]-sample-1][time][k] = (axis[k])[sample][time]
        test_label[test_x.shape[0]-sample-1] = test_l[sample]

    print("Test shape: ", test.shape, "Test label shape: ", test_label.shape)
    return (test, test_label)


def augumentDataset(x_tr, y_tr, infraTimeAcc, infraPerc, dataAugumentationRatio):

    # data augumentation
    blabla = 1
    givenAxisAccDec = False
    givenAxisPerc = 0.2
    augShape = (int(dataAugumentationRatio*x_tr.shape[0]), x_tr.shape[1], x_tr.shape[2])
    print("Adding %d Training Set Entries" % augShape[0])


    # creating augumented np array
    train = np.empty(augShape)
    train_l = np.empty(augShape[0])

    # burst mode
    burst = False

    # populating arrays
    for i in range(0, augShape[0]):
        r = random.random()
        if (infraTimeAcc and r <= infraPerc):
            start = random.randint(10, augShape[1]-20)
            end = random.randint(start+10, augShape[1]-10)
            interval = int(float(end-start)/2)
        else:
            start = 0
            end = augShape[1]
        ii = i % (x_tr.shape[0])

        # fractional part of the acceleration/deceleration modification
        posR = random.random()
        negR = -posR

        # integer part of the acceleration/deceletarion modification
        # acceleration greater than 2 are less likely to happen
        temp = random.random()
        if(temp >= 0.85):
            pintR = random.randint(2,3)
        else:
            pintR = random.randint(0,1)
        nintR = -pintR
        for j in range(0, augShape[1]):
            for k in range(0, augShape[2]):
                if(i%2==0):
                    # adding an 'accellerated (existing) motion' on a portion of time
                    if (infraTimeAcc and r <= infraPerc):
                        if (j >= start and j <=end):
                            if burst:
                                train[i][j][k] = x_tr[ii][j][k]+(posR+pintR)
                            else:
                                if(j <= start+interval):
                                    train[i][j][k] = x_tr[ii][j][k]+(posR+pintR)*(j-start)/float(interval)
                                else:
                                    train[i][j][k] = x_tr[ii][j][k]-(posR+pintR)*((j-end)/float(interval))
                        else:
                            train[i][j][k] = x_tr[ii][j][k]
                    # adding an 'accellerated (existing) motion' on random axis
                    elif (r < givenAxisPerc and givenAxisAccDec):
                        x = random.randint(0,1)
                        y = random.randint(0,1)
                        z = random.randint(0,1)
                        axis = []
                        if(x):
                            axis.append(0)
                        if(y):
                            axis.append(1)
                        if(z):
                            axis.append(2)
                        for a in axis:
                            train[i][j][a] = x_tr[ii][j][a]+(posR+pintR)
                        for k in range(0, 3):
                            if(k not in axis):
                                train[i][j][k] = x_tr[ii][j][k]
                        break
                    # adding an 'accellerated (existing) motion'
                    else:
                        train[i][j][k] = x_tr[ii][j][k]+(posR+pintR)
                else:
                    # adding a 'decellerated (existing) motion'  on a portion of time
                    if (infraTimeAcc and r <= infraPerc):
                        if (j >= start and j <=end):
                            if burst:
                                train[i][j][k] = x_tr[ii][j][k]+(negR+nintR)
                            else:
                                if(j <= start+interval):
                                    train[i][j][k] = x_tr[ii][j][k]+(negR+nintR)*(j-start)/float(interval)
                                else:
                                    train[i][j][k] = x_tr[ii][j][k]-(negR+nintR)*((j-end)/float(interval))
                        else:
                            train[i][j][k] = x_tr[ii][j][k]
                    # adding an 'decellerated (existing) motion' on random axis
                    elif (r < givenAxisPerc and givenAxisPerc):
                        x = random.randint(0,1)
                        y = random.randint(0,1)
                        z = random.randint(0,1)
                        axis = []
                        if(x):
                            axis.append(0)
                        if(y):
                            axis.append(1)
                        if(z):
                            axis.append(2)
                        for a in axis:
                            train[i][j][a] = x_tr[ii][j][a]+(negR+nintR)
                        for k in range(0, 3):
                            if(k not in axis):
                                train[i][j][k] = x_tr[ii][j][k]
                        break
                    # adding an 'decellerated (existing) motion'
                    else:
                        train[i][j][k] = x_tr[ii][j][k]+(negR+nintR)

        # adding label
        train_l[i] = y_tr[ii]

        # debug lines, before after delivery
        """
        if(infraTimeAcc and r <= infraPerc and blabla <= 1 and i%2 == 0):
            if(blabla == 1):
                print()
                for z in range(0, augShape[1]):
                    if (z >= start and z <= end):
                        print("--", z)
                    else:
                        print(z)
                    print(train[i][z][0], train[i][z][1], train[i][z][2])
                    print("(", x_tr[ii][z][0], x_tr[ii][z][1], x_tr[ii][z][2],")")
                print(start, interval, end)
            blabla += 1
        """

    # merging arrays
    x_tr = np.append(x_tr, train, axis=0)
    y_tr = np.append(y_tr, train_l, axis=0)

    print("Data augumented, new shape: ", x_tr.shape)

    return (x_tr, y_tr)


if __name__ == "__main__":
    print("Utility file")
