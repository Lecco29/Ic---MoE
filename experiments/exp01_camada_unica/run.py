#!/usr/bin/env python3
"""Experimento 1 — sensibilidade camada a camada.

Extrai features de cada camada do backbone isoladamente e classifica com kNN
(k=5, euclidiana), sobre 5 folds 70/30. E o que mostra que camadas rasas vao
melhor em cor e camadas profundas em textura.

Gera a Tabela III e a Figura 2 do artigo.

Uso:
  python run.py                                  # todos os backbones, cor e textura
  python run.py --backbone ibot
  python run.py --backbone vgg16 resnet50 --atributo color
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(os.path.dirname(PASTA_SCRIPT))
sys.path.insert(0, PASTA_RAIZ)

from src.backbones import BACKBONES, criar_extrator
from src.dados import N_FOLDS, carregar_fold, extrair_features

PASTA_RESULTADOS = os.path.join(PASTA_SCRIPT, 'results')

# nome da coluna de camada no CSV, por backbone (mantido por compatibilidade
# com os CSVs que ja estavam no repositorio)
COLUNA_CAMADA = {'ibot': 'bloco', 'vmamba': 'stage'}


def avaliar_knn(X_treino, y_treino, X_teste, y_teste, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_treino, y_treino)
    pred = knn.predict(X_teste)
    return (accuracy_score(y_teste, pred) * 100,
            f1_score(y_teste, pred, average='weighted') * 100)


def rodar_atributo(extrator, nome_backbone, atributo, dispositivo):
    """Avalia todas as camadas do backbone nos 5 folds de um atributo."""
    print(f"\n--- {atributo.upper()} ---")

    # por_camada[camada] = {'acc': [...], 'f1': [...], 'dim': int}
    por_camada = {}

    for num_fold in range(1, N_FOLDS + 1):
        print(f"  Fold {num_fold}/{N_FOLDS}...", end=" ", flush=True)

        imgs_tr, rot_tr = carregar_fold(atributo, num_fold, 'train')
        imgs_te, rot_te = carregar_fold(atributo, num_fold, 'test')

        feats_tr = extrair_features(extrator, imgs_tr, nome_backbone, dispositivo)
        feats_te = extrair_features(extrator, imgs_te, nome_backbone, dispositivo)

        acc_do_fold = {}
        for camada in feats_tr:
            acc, f1 = avaliar_knn(feats_tr[camada], rot_tr, feats_te[camada], rot_te)
            dados = por_camada.setdefault(
                camada, {'acc': [], 'f1': [], 'dim': feats_tr[camada].shape[1]})
            dados['acc'].append(acc)
            dados['f1'].append(f1)
            acc_do_fold[camada] = acc

        melhor = max(acc_do_fold, key=acc_do_fold.get)
        print(f"melhor: {melhor}={acc_do_fold[melhor]:.1f}%")

    return por_camada


def salvar(nome_backbone, atributo, por_camada):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    coluna = COLUNA_CAMADA.get(nome_backbone, 'camada')

    # CSV agregado: media e desvio entre os folds
    agregado = pd.DataFrame([
        {
            coluna: camada,
            'accuracy_mean': np.mean(d['acc']),
            'accuracy_std': np.std(d['acc']),
            'f1_score': np.mean(d['f1']),
            'dim': d['dim'],
        }
        for camada, d in por_camada.items() if d['acc']
    ])
    agregado.to_csv(
        os.path.join(PASTA_RESULTADOS, f'{nome_backbone}_{atributo}.csv'), index=False)

    # CSV por fold: usado pelo teste de Wilcoxon
    por_fold = pd.DataFrame([
        {'camada': camada, **{f'fold{i + 1}': acc for i, acc in enumerate(d['acc'])}}
        for camada, d in por_camada.items() if len(d['acc']) == N_FOLDS
    ])
    por_fold.to_csv(
        os.path.join(PASTA_RESULTADOS, f'{nome_backbone}_{atributo}_por_fold.csv'), index=False)

    print(f"\n  {nome_backbone.upper()} {atributo.upper()} (media de {N_FOLDS} folds):")
    for linha in agregado.itertuples(index=False):
        print(f"    {getattr(linha, coluna)}: "
              f"{linha.accuracy_mean:.2f}% (±{linha.accuracy_std:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--backbone', nargs='+', choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument('--atributo', nargs='+', choices=['color', 'texture'],
                        default=['color', 'texture'])
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {dispositivo}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")

    for nome in args.backbone:
        print(f"\n{'=' * 60}\nBACKBONE: {nome.upper()}\n{'=' * 60}")
        try:
            extrator = criar_extrator(nome, dispositivo)
            for atributo in args.atributo:
                salvar(nome, atributo, rodar_atributo(extrator, nome, atributo, dispositivo))
        except Exception as erro:
            # nao aborta a bateria toda se um backbone falhar
            print(f"ERRO com {nome}: {erro}")
            traceback.print_exc()

    print(f"\nFim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
