#!/usr/bin/env python3
"""Curvas de acuracia por profundidade de camada.

Le os CSVs do exp01 e desenha, para cada backbone, a acuracia de cor e de
textura ao longo das camadas — a inversao entre as duas curvas e o resultado
central do artigo. Os melhores pontos vem marcados como E (cor) e D (textura).

Saidas em outputs/figures/:
  curvas_desempenho_profundidade.png   grid 2x2, e a Figura 2 do artigo
  curvas_comparacao_arquiteturas.png   os 4 backbones sobrepostos
  curva_{backbone}.png                 uma figura por backbone

Uso:
  python outputs/gerar_curvas_desempenho.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_SCRIPT)
sys.path.insert(0, PASTA_RAIZ)

from src.backbones import MELHORES_CAMADAS

PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, 'experiments', 'exp01_camada_unica', 'results')
PASTA_FIGURAS = os.path.join(PASTA_SCRIPT, 'figures')

# titulo e rotulo das camadas de cada backbone, na ordem em que aparecem no CSV
BACKBONES = {
    'vgg16':    {'titulo': 'VGG-16 (CNN)', 'prefixo': 'Layer'},
    'resnet50': {'titulo': 'ResNet-50 (CNN)', 'prefixo': 'Layer'},
    'ibot':     {'titulo': 'iBOT (Vision Transformer)', 'prefixo': 'Block'},
    'vmamba':   {'titulo': 'VMamba (State Space Model)', 'prefixo': 'Stage'},
}

COR = {'color': 'red', 'texture': 'blue'}
MARCADOR = {'color': 'o-', 'texture': 's-'}
LEGENDA = {'color': 'Color', 'texture': 'Texture'}

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 18,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 14, 'figure.titlesize': 20,
})


def carregar(backbone):
    """Devolve (rotulos_das_camadas, {atributo: (medias, desvios)})."""
    series, rotulos = {}, None

    for atributo in ('color', 'texture'):
        df = pd.read_csv(os.path.join(PASTA_RESULTADOS, f'{backbone}_{atributo}.csv'))
        series[atributo] = (df.accuracy_mean.to_numpy(), df.accuracy_std.to_numpy())
        if rotulos is None:
            prefixo = BACKBONES[backbone]['prefixo']
            # no CSV a camada vem como 'layer1'/'block0'/'stage1'; exibe 'Layer 1'
            rotulos = [f"{prefixo} {nome[len(prefixo):]}" for nome in df.iloc[:, 0]]

    return rotulos, series


def indice_melhor(backbone, atributo, rotulos):
    """Posicao da melhor camada do artigo dentro da lista de rotulos."""
    prefixo = BACKBONES[backbone]['prefixo']
    nome = MELHORES_CAMADAS[backbone][atributo]
    return rotulos.index(f"{prefixo} {nome[len(prefixo):]}")


def desenhar(ax, backbone, rotulos, series, marcar=True):
    x = np.arange(1, len(rotulos) + 1)

    for atributo in ('color', 'texture'):
        media, desvio = series[atributo]
        ax.plot(x, media, MARCADOR[atributo], color=COR[atributo],
                label=LEGENDA[atributo], linewidth=2)
        ax.fill_between(x, media - desvio, media + desvio,
                        color=COR[atributo], alpha=0.2)

    if marcar:
        for atributo, letra, deslocamento in (('color', 'E', 3), ('texture', 'D', -6)):
            i = indice_melhor(backbone, atributo, rotulos)
            media = series[atributo][0]
            ax.scatter([x[i]], [media[i]], s=220, c=COR[atributo],
                       marker='o' if atributo == 'color' else 's',
                       edgecolors='black', linewidths=2, zorder=5)
            ax.text(x[i] + 0.05, media[i] + deslocamento,
                    f"{letra}={rotulos[i]}", fontsize=13, color=COR[atributo])

    ax.set_title(BACKBONES[backbone]['titulo'], fontweight='bold')
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, rotation=45, ha='right')
    ax.set_ylim(45, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)


def salvar(fig, nome):
    caminho = os.path.join(PASTA_FIGURAS, nome)
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  salvo: {caminho}")


def figura_grid(dados):
    """Figura 2 do artigo: os 4 backbones num grid 2x2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, backbone in zip(axes.flat, BACKBONES):
        rotulos, series = dados[backbone]
        desenhar(ax, backbone, rotulos, series)
    fig.tight_layout()
    salvar(fig, 'curvas_desempenho_profundidade.png')


def figura_comparacao(dados):
    """Backbones sobrepostos, com a profundidade normalizada de 0 a 1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, atributo in zip(axes, ('color', 'texture')):
        for backbone in BACKBONES:
            rotulos, series = dados[backbone]
            media = series[atributo][0]
            # normaliza para comparar backbones com numeros de camadas diferentes
            profundidade = np.linspace(0, 1, len(media))
            ax.plot(profundidade, media, 'o-', label=BACKBONES[backbone]['titulo'].split(' (')[0],
                    linewidth=2, markersize=6)

        ax.set_title(f'{LEGENDA[atributo]} Classification', fontweight='bold')
        ax.set_xlabel('Normalized Depth')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(45, 100)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    salvar(fig, 'curvas_comparacao_arquiteturas.png')


def figuras_individuais(dados):
    for backbone in BACKBONES:
        rotulos, series = dados[backbone]
        fig, ax = plt.subplots(figsize=(8, 5))
        desenhar(ax, backbone, rotulos, series, marcar=False)
        fig.tight_layout()
        salvar(fig, f'curva_{backbone}.png')


def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    dados = {backbone: carregar(backbone) for backbone in BACKBONES}

    figura_grid(dados)
    figura_comparacao(dados)
    figuras_individuais(dados)


if __name__ == '__main__':
    main()
