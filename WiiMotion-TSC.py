import classifierController
from importlib import reload
import classifiers

if __name__ == "__main__":

    while True:
        print("1) Setup dataset")
        print("2) Start Training")
        print("3) Evaluate model on test set")
        print("4) Exit")

        choice = int(input("Your choice: "))

        if(choice == 1):
            reload(classifierController)
            classifierController.setUp(dataAugumentationRatio=8, infraTimeAcc=True, infraPerc=0.2)
        elif(choice == 2):
            try:
                # reloading classifier in case of fast modifications
                reload(classifiers)
                classifierController.startTraining()
            except Exception as e:
                print(str(e))
                print("Error during training")
        elif(choice == 3):
            classifierController.evaluateOnTestSet()
        elif(choice == 4):
            break
        else:
            print("What ?")
