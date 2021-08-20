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
        print("4) Load TestSet from file")
        print("5) Evaluate best model on test set")
        print("6) Start Ensamble Training")
        print("7) Evaluate Ensable model on test set")
        print("8) Save Model to file")
        print("9) Load Best Model")
        print("10) Save Ensamble")
        print("11) Exit")

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
            libraries.classifierController.loadTestSetFromFile()
        elif(choice == 5):
            libraries.classifierController.evaluateOnTestSet()
        elif(choice == 6):
            try:
                # reloading classifier in case of fast modifications
                reload(libraries.classifiers)
                reload(libraries.classifierController)
                libraries.classifierController.ensambleStartTraining()
            except Exception as e:
                traceback.print_exc()
        elif(choice == 7):
            libraries.classifierController.ensambleEvaluate()
        elif(choice == 8):
            libraries.classifierController.saveLastModel()
        elif(choice == 9):
            libraries.classifierController.loadEnsamble()
        elif(choice == 10):
            libraries.classifierController.saveEnsamble()
        elif(choice == 11):
            break
        else:
            print("What ?")
