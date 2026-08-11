"""Carregamento do dataset e extracao de features em lote.

Compartilhado pelos tres experimentos para que todos usem exatamente o mesmo
pre-processamento e o mesmo protocolo de folds.

Os arquivos de fold tem o formato `indice;rotulo;caminho_original`, e o caminho
original aponta para a maquina onde o dataset foi montado. O que importa dele e
so o nome do arquivo: localmente as imagens ficam achatadas numa pasta so, no
padrao `{atributo}_{rotulo}_{arquivo}`.
"""

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PASTA_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = os.path.join(PASTA_RAIZ, 'data')

N_FOLDS = 5

# mesma normalizacao do ImageNet usada no pre-treino dos backbones
TRANSFORMACAO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def caminho_fold(atributo, num_fold, particao):
    """particao: 'train' ou 'test'."""
    return os.path.join(PASTA_DADOS, 'protocols', f'folds_{atributo}_70_30',
                        'folds', f'fold{num_fold}-{particao}.txt')


def pasta_imagens(atributo):
    return os.path.join(PASTA_DADOS, 'images', atributo)


def listar_fold(atributo, num_fold, particao):
    """Le um fold e retorna (caminhos_locais, rotulos), pulando o que nao existe."""
    arquivo = caminho_fold(atributo, num_fold, particao)
    pasta = pasta_imagens(atributo)

    caminhos, rotulos = [], []
    with open(arquivo) as f:
        for linha in f:
            partes = linha.strip().split(';')
            if len(partes) < 3:
                continue

            rotulo = partes[1]
            nome_local = f"{atributo}_{rotulo}_{os.path.basename(partes[2])}"
            caminho = os.path.join(pasta, nome_local)
            if os.path.exists(caminho):
                caminhos.append(caminho)
                rotulos.append(rotulo)

    if not caminhos:
        raise FileNotFoundError(
            f"nenhuma imagem encontrada para {atributo} fold{num_fold}-{particao}; "
            f"confira {pasta}")

    return caminhos, rotulos


def carregar_fold(atributo, num_fold, particao):
    """Le um fold ja como (tensor [N, 3, 224, 224], rotulos).

    Carrega tudo de uma vez na memoria — serve para os exp01/exp02, que so
    fazem inferencia. O exp03 treina com augmentation e usa listar_fold.
    """
    caminhos, rotulos_arquivo = listar_fold(atributo, num_fold, particao)

    imagens, rotulos = [], []
    for caminho, rotulo in zip(caminhos, rotulos_arquivo):
        try:
            imagens.append(TRANSFORMACAO(Image.open(caminho).convert('RGB')))
            rotulos.append(rotulo)
        except OSError:
            continue  # imagem corrompida: pula junto com o rotulo

    if not imagens:
        raise FileNotFoundError(
            f"nenhuma imagem legivel para {atributo} fold{num_fold}-{particao}")

    return torch.stack(imagens), rotulos


def extrair_features(extrator, imagens, nome_backbone, dispositivo, tamanho_lote=32):
    """Roda o extrator em lotes e retorna {camada: array [N, D]}.

    O iBOT usa o token CLS; os demais, GAP sobre o mapa de ativacao.
    """
    acumulado = {}

    for i in range(0, len(imagens), tamanho_lote):
        lote = imagens[i:i + tamanho_lote].to(dispositivo)

        if nome_backbone == 'ibot':
            feats = extrator.extrairFeaturesComCLS(lote)
        else:
            feats = extrator.extrairFeatures(lote, aplicarGAP=True)

        for nome, feat in feats.items():
            acumulado.setdefault(nome, []).append(feat.cpu().numpy())

    return {nome: np.concatenate(partes, axis=0) for nome, partes in acumulado.items()}
