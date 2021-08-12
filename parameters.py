import classifiers

BATCH_SIZE=50
AUGMENT=0
EPOCH=500
SEED=42
KFOLD_SPLIT = 3
PATIENCE = 10
# good ones
#FUNC_NAME = classifiers.simple_mlp
#FUNC_NAME = classifiers.simple_dnn
#FUNC_NAME = classifiers.super_simple_mlp

# sucking models
#FUNC_NAME = classifiers.simple_mlp_experimental
#FUNC_NAME = classifiers.hybrid_restnet
#FUNC_NAME = classifiers.shallow_cnn
#FUNC_NAME = classifiers.get_cnn_standard
#FUNC_NAME = classifiers.rest_net
FUNC_NAME = classifiers.get_cnn_experimental