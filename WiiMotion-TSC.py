import libraries.classifierController as cc
from importlib import reload
import libraries.classifiers
import traceback
from libraries.parameters import *

if __name__ == "__main__":

    while True:
        print("1) Setup dataset")
        print("2) Setup dataset (randomize test selection)")
        print("3) Start Training")
        print("4) Evaluate model on test set")
        print("5) Save Model to file")
        print("6) Exit")

        try:
            choice = int(input("Your choice: "))
        except:
            print("What ?")
            continue

        if(choice == 1):
            reload(libraries.classifierController)
            libraries.classifierController.setUp(dataAugumentationRatio=AUGMENT, infraTimeAcc=False, infraPerc=0.1)
        elif(choice == 2):
            reload(libraries.classifierController)
            libraries.classifierController.setUp(dataAugumentationRatio=AUGMENT, infraTimeAcc=False, infraPerc=0.1, random=1, seed=SEED, approx=0)    
        elif(choice == 3):
            try:
                # reloading classifier in case of fast modifications
                reload(libraries.classifiers)
                reload(libraries.classifierController)
                libraries.classifierController.startTraining()
            except Exception as e:
                traceback.print_exc()
        elif(choice == 4):
            libraries.classifierController.evaluateOnTestSet()
        elif(choice == 5):
            libraries.classifierController.saveLastModel()
        elif(choice == 6):
            break
        else:
            print("What ?")
