# Configs -> TestSet Evaluation

1. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> 0.957
21. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> 0.948
22. dataAugumentationRatio=14, infraTimeAcc=True(2-5), infraPerc=0.1, brust=False, batch_size=100, simple_mlp -> ...
2. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_mlp -> 0.959         BEST
3. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_mlp -> 0.931
4. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, super_simple_mlp -> 0.948
5. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, super_simple_mlp -> 0.926
51. dataAugumentationRatio=14, infraTimeAcc=True, infraPerc=0.1, brust=False, batch_size=100, super_simple_mlp(315*3) -> 0.944
6. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=100, simple_ddn ->
7. dataAugumentationRatio=14, infraTimeAcc=False, infraPerc=any, brust=any, batch_size=500, simple_ddn ->

Provare test 21. aggiungendo accellerazione tra 2 e 5 (e non solo tra 2 e 3).
Moltiplicare i layer del simplemlp per 3 (con la iglior configurazione).
Testare i ddn.
Scelto il miogliore qui, vedere se cambia qualcosa aumentando il dataAugRatio a 20.
Provare poi a cambiare learning rate e momentum.
