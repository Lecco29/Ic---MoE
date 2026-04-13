# avaliacao das features usando classificador knn
# usa o protocolo 70/30 com folds pre-definidos
# calcula acuracia e f1-score pra cada camada

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score


def avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste, k=5):
    # sklearn precisa de labels numericos
    codificador = LabelEncoder()
    yTreino = codificador.fit_transform(labelsTreino)
    yTeste = codificador.transform(labelsTeste)

    resultados = {}

    for nomeCamada in featuresTreino.keys():
        xTreino = featuresTreino[nomeCamada]
        xTeste = featuresTeste[nomeCamada]

        classificador = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        classificador.fit(xTreino, yTreino)
        predicoes = classificador.predict(xTeste)

        acuracia = accuracy_score(yTeste, predicoes) * 100
        f1 = f1_score(yTeste, predicoes, average='weighted') * 100

        resultados[nomeCamada] = {
            'accuracy': acuracia,
            'f1_score': f1,
            'dim': xTreino.shape[1]
        }

    return resultados
