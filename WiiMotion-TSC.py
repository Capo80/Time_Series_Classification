import classifierController
from importlib import reload
import classifiers
import traceback

if __name__ == "__main__":

    while True:
        print("1) Setup dataset")
        print("2) Setup dataset (randomize test selection)")
        print("3) Start Training")
        print("4) Evaluate model on test set")
        print("5) Exit")

        try:
            choice = int(input("Your choice: "))
        except:
            print("What ?")
            continue

        if(choice == 1):
            reload(classifierController)
            classifierController.setUp(dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=0.1)
        elif(choice == 2):
            try:
                seed = int(input("Choose seed: "))
            except:
                print("What ?")
                continue

            reload(classifierController)
            classifierController.setUp(dataAugumentationRatio=0, infraTimeAcc=False, infraPerc=0.1, random=1, seed=seed, approx=0)    
        elif(choice == 3):
            try:
                # reloading classifier in case of fast modifications
                reload(classifiers)
                reload(classifierController)
                classifierController.startTraining()
            except Exception as e:
                traceback.print_exc()
        elif(choice == 4):
            classifierController.evaluateOnTestSet()
        elif(choice == 5):
            break
        else:
            print("What ?")
