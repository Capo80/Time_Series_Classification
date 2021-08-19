import libraries.classifierController
import libraries.classifiers
import libraries.parameters
from libraries.constants import *
import random,string,os,time
from importlib import reload

def train_and_evaluate():
    libraries.classifierController.setUp(dataAugumentationRatio=libraries.parameters.AUGMENT, infraTimeAcc=False, infraPerc=0.1, random=1, seed=libraries.parameters.SEED, approx=0)
    reload(libraries.classifiers)
    reload(libraries.classifierController)
    libraries.classifierController.startTraining()
    return libraries.classifierController.evaluateOnTestSet()

best_evaluation = [1000, 0]
best_model = None
best_eva_evaluation = [0, 0]
best_eva_model = None

for j in range(450, 500, 50):
	for i in range(0, 20, 5):

		libraries.parameters.AUGMENT = i
		libraries.parameters.BATCH_SIZE = j
		curr_evaluation, curr_model = train_and_evaluate()

		if (curr_evaluation[0] < best_evaluation[0]):
			best_evaluation = curr_evaluation
			best_model = curr_model

		if (curr_evaluation[1] > best_eva_evaluation[1]):
			best_eva_evaluation = curr_evaluation
			best_eva_model = curr_model
		


print("Best Model at the end is: ", best_evaluation)
print("Best Eva Model at the end is: ", best_eva_evaluation)
random.seed(time.time())
folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
print(folder)
os.mkdir(models_path + "/" + folder)
best_model.save(models_path + "/" + folder)

folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
print(folder)
os.mkdir(models_path + "/" + folder)
best_eva_model.save(models_path + "/" + folder)