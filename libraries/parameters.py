import libraries.classifiers as cc

BATCH_SIZE=100
AUGMENT=14
EPOCH=100
SEED=123456789
KFOLD_SPLIT = 5
PATIENCE = 10
STRATIFIED = True
SYNT = True
INCR = True
TO_INCREASE = [20, 30, 90, 70, 50, 100, 10, 50]
#TO_INCREASE = [40, 30, 70, 60, 100, 70, 100, 0]

# good ones
FUNC_NAME = cc.simple_mlp
#FUNC_NAME = cc.eight
#FUNC_NAME = cc.simple_dnn
#FUNC_NAME = cc.super_simple_mlp

# sucking models
#FUNC_NAME = cc.simple_mlp_experimental2
#FUNC_NAME = cc.hybrid_restnet
#FUNC_NAME = cc.shallow_cnn
#FUNC_NAME = cc.get_cnn_standard
#FUNC_NAME = cc.rest_net
#FUNC_NAME = cc.get_cnn_experimental
