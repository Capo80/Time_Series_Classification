import libraries.classifiers as cc

BATCH_SIZE=150
AUGMENT=20
EPOCH=200
SEED=42
KFOLD_SPLIT = 5
PATIENCE = 10
# good ones
#FUNC_NAME = cc.simple_mlp
FUNC_NAME = cc.simple_dnn
#FUNC_NAME = cc.super_simple_mlp

# sucking models
#FUNC_NAME = cc.simple_mlp_experimental2
#FUNC_NAME = cc.hybrid_restnet
#FUNC_NAME = cc.shallow_cnn
#FUNC_NAME = cc.get_cnn_standard
#FUNC_NAME = cc.rest_net
#FUNC_NAME = cc.get_cnn_experimental
