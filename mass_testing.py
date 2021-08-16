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

for i in range(0, 20):

	libraries.parameters.AUGMENT = i
	curr_evaluation, curr_model = train_and_evaluate()

	if (curr_evaluation[0] < best_evaluation[0]):
		best_evaluation = curr_evaluation
		best_model = curr_model

print("Best Model at the end is: ", best_evaluation)
random.seed(time.time())
folder = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
os.mkdir(models_path + "/" + folder)
best_model.save(models_path + "/" + folder)
