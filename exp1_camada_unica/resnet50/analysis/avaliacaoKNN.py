# avaliacao com knn - acuracia e f1 por camada

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch


class AvaliadorKNN:

    def __init__(self, k=5):
        self.k = k

    def avaliar(self, featuresTreino, labelsTreino, featuresTeste, labelsTeste):
        # sklearn precisa de numpy
        if isinstance(featuresTreino, torch.Tensor):
            featuresTreino = featuresTreino.numpy()
        if isinstance(featuresTeste, torch.Tensor):
            featuresTeste = featuresTeste.numpy()

        classificador = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')
        classificador.fit(featuresTreino, labelsTreino)
        predicoes = classificador.predict(featuresTeste)

        acuracia = accuracy_score(labelsTeste, predicoes) * 100
        f1 = f1_score(labelsTeste, predicoes, average='weighted') * 100

        return {'accuracy': acuracia, 'f1_score': f1, 'predicoes': predicoes}

    def avaliarMultiplasCamadas(self, featuresDictTreino, labelsTreino,
                                 featuresDictTeste, labelsTeste, dimensoes):
        resultados = {}
        for nomeCamada in featuresDictTreino.keys():
            res = self.avaliar(featuresDictTreino[nomeCamada], labelsTreino,
                               featuresDictTeste[nomeCamada], labelsTeste)
            resultados[nomeCamada] = {
                'accuracy': res['accuracy'],
                'f1_score': res['f1_score'],
                'dim': dimensoes[nomeCamada]
            }
        return resultados


def criarAvaliador(k=5):
    return AvaliadorKNN(k=k)
