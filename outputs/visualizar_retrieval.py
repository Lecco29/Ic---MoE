#!/usr/bin/env python3
"""Figuras qualitativas de retrieval: query -> top-K vizinhos.

Cada linha da figura e uma query e seus K vizinhos mais proximos no conjunto de
treino, com borda verde quando a classe bate e vermelha quando erra. Usa o fold 1
e a melhor camada de cada backbone segundo o exp01.

Gera as Figuras 3 e 4 do artigo (iBOT, block2 para cor e block9 para textura).

Uso:
  python outputs/visualizar_retrieval.py                    # todos os backbones
  python outputs/visualizar_retrieval.py --backbone ibot
  python outputs/visualizar_retrieval.py --backbone ibot --atributo color --k 5
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_SCRIPT)
sys.path.insert(0, PASTA_RAIZ)

from src.backbones import BACKBONES, MELHORES_CAMADAS, criar_extrator
from src.dados import TRANSFORMACAO, listar_fold
from src.evaluation.retrieval import computar_metricas_retrieval

PASTA_FIGURAS = os.path.join(PASTA_SCRIPT, 'figures', 'retrieval')

TAMANHO_MINIATURA = (112, 112)
VERDE, VERMELHO, CINZA = '#2E7D32', '#C62828', '#333333'


def carregar_imagens(caminhos):
    """Le e pre-processa uma lista de caminhos, na mesma ordem."""
    return torch.stack([TRANSFORMACAO(Image.open(c).convert('RGB')) for c in caminhos])


def extrair_camada(extrator, imagens, nome_backbone, dispositivo, camada, tamanho_lote=32):
    """Extrai features de uma unica camada, em lotes."""
    partes = []
    for i in range(0, len(imagens), tamanho_lote):
        lote = imagens[i:i + tamanho_lote].to(dispositivo)
        if nome_backbone == 'ibot':
            feats = extrator.extrairFeaturesComCLS(lote)
        else:
            feats = extrator.extrairFeatures(lote, aplicarGAP=True)
        partes.append(feats[camada].cpu().numpy())
    return np.concatenate(partes, axis=0)


def selecionar_queries(rotulos, n_por_classe=1, seed=42):
    """Sorteia n imagens de cada classe para servirem de query."""
    rng = np.random.default_rng(seed)
    indices = []
    for classe in sorted(set(rotulos)):
        da_classe = [i for i, r in enumerate(rotulos) if r == classe]
        escolhidos = rng.choice(da_classe, size=min(n_por_classe, len(da_classe)),
                                replace=False)
        indices.extend(escolhidos.tolist())
    return indices


def miniatura(caminho):
    return np.array(Image.open(caminho).convert('RGB').resize(TAMANHO_MINIATURA,
                                                              Image.LANCZOS))


def _formatar(ax, imagem, cor_borda, titulo, rotulo):
    ax.imshow(imagem)
    ax.set_xticks([])
    ax.set_yticks([])
    for borda in ax.spines.values():
        borda.set_edgecolor(cor_borda)
        borda.set_linewidth(3)
    if titulo:
        ax.set_title(titulo, fontsize=7, pad=2)
    ax.set_xlabel(rotulo, fontsize=6, labelpad=2)


def gerar_figura(indices_query, caminhos_teste, rotulos_teste, ranking,
                 caminhos_treino, rotulos_treino, nome_backbone, atributo, camada, k):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)

    n_linhas, n_colunas = len(indices_query), k + 1
    fig, axes = plt.subplots(n_linhas, n_colunas,
                             figsize=(n_colunas * 1.4, n_linhas * 1.6))
    if n_linhas == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'{nome_backbone.upper()} — {atributo.upper()} — camada {camada} — top-{k} retrieval',
        fontsize=10, fontweight='bold', y=1.01)

    for linha, indice_query in enumerate(indices_query):
        classe_query = rotulos_teste[indice_query]
        titulo = 'Query' if linha == 0 else None
        _formatar(axes[linha, 0], miniatura(caminhos_teste[indice_query]),
                  CINZA, titulo, classe_query)

        for coluna, indice_vizinho in enumerate(ranking[indice_query, :k], start=1):
            classe_vizinho = rotulos_treino[indice_vizinho]
            _formatar(axes[linha, coluna], miniatura(caminhos_treino[indice_vizinho]),
                      VERDE if classe_vizinho == classe_query else VERMELHO,
                      f'R{coluna}' if linha == 0 else None, classe_vizinho)

    fig.legend(handles=[mpatches.Patch(facecolor=VERDE, label='Correto'),
                        mpatches.Patch(facecolor=VERMELHO, label='Errado')],
               loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()

    base = f'{nome_backbone}_{atributo}_retrieval_top{k}'
    for extensao in ('pdf', 'png'):
        caminho = os.path.join(PASTA_FIGURAS, f'{base}.{extensao}')
        plt.savefig(caminho, dpi=150, bbox_inches='tight')
        print(f"  salvo: {caminho}")
    plt.close()


def rodar(nome_backbone, atributo, dispositivo, k, fold=1):
    print(f"\n{'=' * 55}")
    print(f"{nome_backbone.upper()} — {atributo.upper()} (fold {fold}, top-{k})")
    print(f"{'=' * 55}")

    camada = MELHORES_CAMADAS[nome_backbone][atributo]
    print(f"  camada: {camada}")

    caminhos_tr, rotulos_tr = listar_fold(atributo, fold, 'train')
    caminhos_te, rotulos_te = listar_fold(atributo, fold, 'test')
    print(f"  treino: {len(caminhos_tr)} | teste: {len(caminhos_te)}")

    extrator = criar_extrator(nome_backbone, dispositivo)
    feats_tr = extrair_camada(extrator, carregar_imagens(caminhos_tr),
                              nome_backbone, dispositivo, camada)
    feats_te = extrair_camada(extrator, carregar_imagens(caminhos_te),
                              nome_backbone, dispositivo, camada)

    # ranking por distancia euclidiana: cada query contra toda a galeria de treino
    ranking = np.argsort(euclidean_distances(feats_te, feats_tr), axis=1)

    metricas = computar_metricas_retrieval(feats_tr, rotulos_tr, feats_te, rotulos_te)
    print(f"  mAP@10={metricas['map_at_10']:.2f}%  "
          f"R@1={metricas['r_at_1']:.2f}%  R@5={metricas['r_at_5']:.2f}%")

    indices_query = selecionar_queries(rotulos_te)
    print(f"  {len(indices_query)} queries ({len(set(rotulos_te))} classes)")

    gerar_figura(indices_query, caminhos_te, rotulos_te, ranking,
                 caminhos_tr, rotulos_tr, nome_backbone, atributo, camada, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--backbone', nargs='+', choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument('--atributo', nargs='+', choices=['color', 'texture'],
                        default=['color', 'texture'])
    parser.add_argument('--k', type=int, default=5, help='quantos vizinhos exibir')
    parser.add_argument('--fold', type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {dispositivo}")

    for backbone in args.backbone:
        for atributo in args.atributo:
            rodar(backbone, atributo, dispositivo, args.k, args.fold)

    print(f"\nFiguras em: {PASTA_FIGURAS}")


if __name__ == '__main__':
    main()
