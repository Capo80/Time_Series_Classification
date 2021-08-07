# Configs -> TestSet Evaluation

1. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> 0.957
12. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> 0.948
13. dataAugumentationRatio=14, infraTimeAcc=True(2-5), infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> 0.939
2. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp -> 0.959
21. dataAugumentationRatio=14, infraTimeAcc=True(2-5), infraPerc=any, brust=any, batch_size=100, simple_mlp -> 0.946
22. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp(param*3) -> 0.953
23. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp -> 0.954
231. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp(3layer) -> 0.963
    BEST (n_slit = 5)
232. dataAugumentationRatio=30, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp(3layer) -> 0.955
233. dataAugumentationRatio=8, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp(3layer) -> 0.960
    BEST
24. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=250, simple_mlp -> 0.948
3. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_mlp -> 0.931
4. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, super_simple_mlp -> 0.948
5. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, super_simple_mlp -> 0.926
51. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, super_simple_mlp(315*3) -> 0.944
6. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_ddn(5, 3*315 ... 50) -> 0.958
7. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_ddn(5, 315,315...) -> 0.960
8. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_ddn(g) -> 0.944
9. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_ddn(p) -> 0.953
10. dataAugumentationRatio=20, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_ddn(a) -> 0.949

p: 347, 499, 112 - 440, 350, 210, 409, 327
a: 356, 112, 415 - 109, 206, 230, 360, 385
g: 6, 288, 35    - 100, 56, 89, 455, 347

Provare a cambiare learning rate e momentum.
