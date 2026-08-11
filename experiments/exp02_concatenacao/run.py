#!/usr/bin/env python3
"""Experimento 2 — fusao por concatenacao.

Concatena a melhor camada de cor (E) com a melhor de textura (D), Z = [E || D],
e compara com usar cada uma sozinha. A conclusao do artigo e que concatenar
geralmente nao ajuda: em cor chega a piorar bastante (iBOT cai ~21 pp).

Gera a Tabela V e as linhas "Concat." da Tabela VIII.

Uso:
  python run.py                                  # todos os backbones
  python run.py --backbone ibot
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

from src.backbones import BACKBONES, MELHORES_CAMADAS, criar_extrator
from src.dados import N_FOLDS, carregar_fold, extrair_features
from src.evaluation.retrieval import computar_metricas_retrieval

PASTA_RESULTADOS = os.path.join(PASTA_SCRIPT, 'results')

METRICAS_RETRIEVAL = ('map_at_10', 'r_at_1', 'r_at_5')


def camadas_do_backbone(nome_backbone):
    """E = camada rasa (melhor em cor); D = camada profunda (melhor em textura)."""
    melhores = MELHORES_CAMADAS[nome_backbone]
    return {'E': melhores['color'], 'D': melhores['texture']}


def avaliar_knn(X_treino, y_treino, X_teste, y_teste, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_treino, y_treino)
    pred = knn.predict(X_teste)
    return (accuracy_score(y_teste, pred) * 100,
            f1_score(y_teste, pred, average='macro') * 100)


def media_retrieval(medidas):
    """Media de cada metrica de retrieval ao longo dos folds."""
    return {m: np.mean([d[m] for d in medidas]) for m in METRICAS_RETRIEVAL}


def rodar_atributo(extrator, nome_backbone, atributo, dispositivo):
    print(f"\n--- {atributo.upper()} ---")
    camadas = camadas_do_backbone(nome_backbone)

    # acumula por representacao: E (rasa), D (profunda), Z (concatenada)
    acc = {'E': [], 'D': [], 'Z': []}
    f1 = {'E': [], 'D': [], 'Z': []}
    retrieval = {'E': [], 'D': [], 'Z': []}
    por_fold = []

    for num_fold in range(1, N_FOLDS + 1):
        print(f"  Fold {num_fold}/{N_FOLDS}...", end=" ", flush=True)

        imgs_tr, rot_tr = carregar_fold(atributo, num_fold, 'train')
        imgs_te, rot_te = carregar_fold(atributo, num_fold, 'test')

        feats_tr = extrair_features(extrator, imgs_tr, nome_backbone, dispositivo)
        feats_te = extrair_features(extrator, imgs_te, nome_backbone, dispositivo)

        treino = {'E': feats_tr[camadas['E']], 'D': feats_tr[camadas['D']]}
        teste = {'E': feats_te[camadas['E']], 'D': feats_te[camadas['D']]}
        treino['Z'] = np.concatenate([treino['E'], treino['D']], axis=1)
        teste['Z'] = np.concatenate([teste['E'], teste['D']], axis=1)

        for rep in ('E', 'D', 'Z'):
            a, f = avaliar_knn(treino[rep], rot_tr, teste[rep], rot_te)
            acc[rep].append(a)
            f1[rep].append(f)
            retrieval[rep].append(
                computar_metricas_retrieval(treino[rep], rot_tr, teste[rep], rot_te))

        por_fold.append({'fold': num_fold, 'acc_E': acc['E'][-1],
                         'acc_D': acc['D'][-1], 'acc_Z': acc['Z'][-1]})
        print(f"E={acc['E'][-1]:.1f}%  D={acc['D'][-1]:.1f}%  Z={acc['Z'][-1]:.1f}%")

    medias = {rep: media_retrieval(retrieval[rep]) for rep in ('E', 'D', 'Z')}

    resultado = {'backbone': nome_backbone, 'atributo': atributo}
    for rep in ('E', 'D', 'Z'):
        if rep != 'Z':
            resultado[f'camada_{rep}'] = camadas[rep]
        resultado[f'acc_{rep}_media'] = np.mean(acc[rep])
        resultado[f'acc_{rep}_std'] = np.std(acc[rep])
        resultado[f'f1_{rep}_media'] = np.mean(f1[rep])
        resultado[f'map10_{rep}'] = medias[rep]['map_at_10']
        resultado[f'r1_{rep}'] = medias[rep]['r_at_1']
        resultado[f'r5_{rep}'] = medias[rep]['r_at_5']

    # quanto a concatenacao ganhou (ou perdeu) sobre a melhor camada isolada
    resultado['melhora'] = (resultado['acc_Z_media']
                            - max(resultado['acc_E_media'], resultado['acc_D_media']))

    print(f"\n  Resultado {atributo}:")
    for rep, rotulo in (('E', f"E ({camadas['E']})"), ('D', f"D ({camadas['D']})"),
                        ('Z', 'Z (fusao)')):
        print(f"    {rotulo:<16} acc={resultado[f'acc_{rep}_media']:6.2f}%  "
              f"mAP@10={resultado[f'map10_{rep}']:6.2f}%  "
              f"R@1={resultado[f'r1_{rep}']:6.2f}%  R@5={resultado[f'r5_{rep}']:6.2f}%")
    print(f"    melhora da fusao: {resultado['melhora']:+.2f} pp")

    return resultado, por_fold


def salvar(resultados, linhas_por_fold):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    caminho = os.path.join(PASTA_RESULTADOS, 'fusao.csv')
    pd.DataFrame(resultados).to_csv(caminho, index=False)
    print(f"\nResultados salvos em: {caminho}")

    if linhas_por_fold:
        caminho_fold = os.path.join(PASTA_RESULTADOS, 'fusao_por_fold.csv')
        pd.DataFrame(linhas_por_fold).to_csv(caminho_fold, index=False)
        print(f"Por-fold salvo em:   {caminho_fold}")


def imprimir_resumo(resultados):
    print(f"\n{'=' * 78}")
    print(f"RESUMO — EXP02: CONCATENACAO EARLY+DEEP (70/30, {N_FOLDS} folds)")
    print(f"{'=' * 78}")
    print(f"{'Backbone':<12} {'Atrib.':<9} {'Metodo':<12} "
          f"{'Acc(%)':>7} {'mAP@10':>8} {'R@1':>7} {'R@5':>7}")
    print("-" * 78)

    for r in resultados:
        for rep, rotulo in (('E', 'Early (E)'), ('D', 'Deep (D)'), ('Z', 'Concat (Z)')):
            print(f"{r['backbone']:<12} {r['atributo']:<9} {rotulo:<12} "
                  f"{r[f'acc_{rep}_media']:>7.2f} {r[f'map10_{rep}']:>8.2f} "
                  f"{r[f'r1_{rep}']:>7.2f} {r[f'r5_{rep}']:>7.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--backbone', nargs='+', choices=BACKBONES, default=list(BACKBONES))
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {dispositivo}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")

    resultados, linhas_por_fold = [], []
    for nome in args.backbone:
        print(f"\n{'=' * 60}\nBACKBONE: {nome.upper()}\n{'=' * 60}")
        try:
            extrator = criar_extrator(nome, dispositivo)
            for atributo in ('color', 'texture'):
                resultado, por_fold = rodar_atributo(extrator, nome, atributo, dispositivo)
                resultados.append(resultado)
                linhas_por_fold.extend(
                    {'backbone': nome, 'atributo': atributo, **linha} for linha in por_fold)
        except Exception as erro:
            # nao aborta a bateria toda se um backbone falhar
            print(f"ERRO com {nome}: {erro}")
            traceback.print_exc()

    if not resultados:
        print("nenhum resultado gerado")
        return

    salvar(resultados, linhas_por_fold)
    imprimir_resumo(resultados)
    print(f"Fim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
