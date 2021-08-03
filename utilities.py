import numpy as np
from constants import *

def defineArchitecture(model, modelCode):

    if(modelCode == ARCH1):
        import file_arch1
    elif(modelCode == ARCH2):
        import file_arch2
    else:
        print("Model Code not found")
        exit(-1)


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


def getTestSet(filenameX=None, filenameY=None, filenameZ=None, filnameL=None, perc=0.3):

    if(filenameX != None and filenameY != None and filenameZ != None and filenameL != None):
        test_x = np.loadtxt(open(filenameX, "rb"), delimiter=",")
        test_y = np.loadtxt(open(filenameY, "rb"), delimiter=",")
        test_z = np.loadtxt(open(filenameZ, "rb"), delimiter=",")
        test_l = np.loadtxt(open(filenameL, "rb"), delimiter=",")
    else:
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

if __name__ == "__main__":
    getTestSet()
    getTrainingSet()
