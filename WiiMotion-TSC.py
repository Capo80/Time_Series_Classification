import libraries.classifierController as cc
from importlib import reload
import libraries.classifiers
import traceback
from libraries.parameters import *

if __name__ == "__main__":

    while True:
        print("|----------------DATA OPS----------------------")
        print("1) Setup dataset")
        print("2) Setup dataset (with shuffle)")
        print("-----------------TRAIN OPS---------------------")
        print("3) Start Training")
        print("4) Start Ensamble Training")
        print("-----------------EVAL OPS----------------------")
        print("5) Evaluate Ensable model on test set")
        print("6) Evaluate Model on test set")
        print("7) Evaluate Ensamble Max Model on test set")
        print("-----------------LOAD OPS----------------------")
        print("8) Load TestSet from file")
        print("9) Load Best Model")
        print("10) Load Best Ensemble")
        print("-----------------SAVE OPS----------------------")
        print("11) Save Current Model")
        print("12) Save Ensamble")
        print("-----------------------------------------------")
        print("13) Exit")
        print("|----------------------------------------------")

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
        elif(choice == 8):
            libraries.classifierController.loadTestSetFromFile()
        elif(choice == 6):
            libraries.classifierController.evaluateOnTestSet()
        elif(choice == 4):
            try:
                # reloading classifier in case of fast modifications
                reload(libraries.classifiers)
                reload(libraries.classifierController)
                libraries.classifierController.ensambleStartTraining()
            except Exception as e:
                traceback.print_exc()
        elif(choice == 5):
            libraries.classifierController.ensambleEvaluate()
        elif(choice == 7):
            libraries.classifierController.ensambleEvaluateMax()
        elif(choice == 11):
            libraries.classifierController.saveLastModel()
        elif(choice == 9):
            libraries.classifierController.loadBestModel()
        elif(choice == 10):
            libraries.classifierController.loadEnsamble()
        elif(choice == 12):
            libraries.classifierController.saveEnsamble()
        elif(choice == 13):
            break
        else:
            print("What ?")
