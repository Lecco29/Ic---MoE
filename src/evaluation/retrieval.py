import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def computar_metricas_retrieval(X_treino, y_treino, X_teste, y_teste, K=10):
    """
    Para cada imagem de teste, rankeia as imagens de treino por distancia euclidiana
    e calcula metricas de recuperacao tipicas de CBIR.

    Retorna dict com map_at_10, r_at_1 e r_at_5 (valores em %, 0-100).
    """
    y_treino = np.array(y_treino)
    y_teste  = np.array(y_teste)

    # (n_teste, n_treino) — distancias euclidianas
    dist = euclidean_distances(X_teste, X_treino)
    ranking = np.argsort(dist, axis=1)  # indices do mais proximo ao mais distante

    ap_list, r1_list, r5_list = [], [], []

    for i in range(len(y_teste)):
        classe  = y_teste[i]
        top_k   = y_treino[ranking[i, :K]]

        # R@1: o primeiro resultado e da classe correta?
        r1_list.append(float(top_k[0] == classe))

        # R@5: a classe correta aparece nos 5 primeiros?
        r5_list.append(float(classe in top_k[:5]))

        # mAP@K: precisao media ponderada pelos itens relevantes
        relevante   = (top_k == classe).astype(float)
        n_relevante = relevante.sum()
        if n_relevante == 0:
            ap_list.append(0.0)
            continue
        precisao_acum = np.cumsum(relevante) / np.arange(1, K + 1)
        ap = np.sum(precisao_acum * relevante) / min(n_relevante, K)
        ap_list.append(ap)

    return {
        'map_at_10': np.mean(ap_list) * 100,
        'r_at_1':    np.mean(r1_list) * 100,
        'r_at_5':    np.mean(r5_list) * 100,
    }
