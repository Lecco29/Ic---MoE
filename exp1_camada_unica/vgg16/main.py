#!/usr/bin/env python3
# experimento VGG-16 - classificacao de atributos visuais em roupas
# usando protocolo de validacao 70/30 com 5 folds

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from torchvision import transforms

# configura os caminhos do projeto pra importar os modulos
PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA_RAIZ)
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'models'))
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'features'))
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'analysis'))

from carregadorVGG import criarExtrator
from extracaoFeatures import extrairFeatures
from avaliacaoKNN import avaliarKNNComFolds


# transformacoes padrao pra imagens do imagenet
# redimensiona pra 224x224 e normaliza com a media e desvio padrao do imagenet
TRANSFORMACAO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def carregarImagensDoFold(arquivoFold, pastaBase, tipoAtributo):
    # formato do arquivo: indice;classe;caminho_original
    listaImagens = []
    listaLabels = []

    with open(arquivoFold, 'r') as f:
        linhas = f.readlines()

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        partes = linha.split(';')
        if len(partes) < 3:
            continue

        classe = partes[1]
        caminhoOriginal = partes[2]

        # os arquivos locais seguem o padrao: tipoAtributo_classe_nomeOriginal
        partesPath = caminhoOriginal.split('/')
        subpasta = partesPath[-2]
        nomeArquivoOriginal = partesPath[-1]
        nomeArquivoLocal = f"{tipoAtributo}_{subpasta}_{nomeArquivoOriginal}"
        caminhoLocal = os.path.join(pastaBase, nomeArquivoLocal)

        if os.path.exists(caminhoLocal):
            try:
                img = Image.open(caminhoLocal).convert('RGB')
                listaImagens.append(TRANSFORMACAO(img))
                listaLabels.append(classe)
            except Exception as e:
                print(f"erro ao carregar imagem: {caminhoLocal}")

    if len(listaImagens) == 0:
        return None, None

    return torch.stack(listaImagens), listaLabels


def rodarExperimentoComFolds(nome, pastaFolds, pastaImagens, modelo, device, tipoAtributo):
    print(f"\n[{nome}]")

    # vgg16 tem 5 camadas de pooling
    dadosPorCamada = {f'layer{i}': {'acc': [], 'f1': [], 'dim': 0} for i in range(1, 6)}

    for numFold in range(1, 6):
        print(f"\n  processando fold {numFold}/5:")

        arquivoTreino = os.path.join(pastaFolds, f'fold{numFold}-train.txt')
        arquivoTeste = os.path.join(pastaFolds, f'fold{numFold}-test.txt')

        if not os.path.exists(arquivoTreino) or not os.path.exists(arquivoTeste):
            print(f"    arquivos do fold nao encontrados!")
            continue

        imagensTreino, labelsTreino = carregarImagensDoFold(arquivoTreino, pastaImagens, tipoAtributo)
        if imagensTreino is None:
            print(f"    erro ao carregar imagens de treino")
            continue
        print(f"    treino: {len(imagensTreino)} | ", end="")

        imagensTeste, labelsTeste = carregarImagensDoFold(arquivoTeste, pastaImagens, tipoAtributo)
        if imagensTeste is None:
            print(f"erro ao carregar imagens de teste")
            continue
        print(f"teste: {len(imagensTeste)}")

        featuresTreino = extrairFeatures(modelo, imagensTreino, device)
        featuresTeste = extrairFeatures(modelo, imagensTeste, device)
        resultados = avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste)

        for camada, res in resultados.items():
            dadosPorCamada[camada]['acc'].append(res['accuracy'])
            dadosPorCamada[camada]['f1'].append(res['f1_score'])
            dadosPorCamada[camada]['dim'] = res['dim']

        melhorCamada = max(resultados.keys(), key=lambda c: resultados[c]['accuracy'])
        print(f"    melhor: {melhorCamada} = {resultados[melhorCamada]['accuracy']:.2f}%")

    print(f"\nresultados finais {nome} (media dos 5 folds):")
    resultadoFinal = {}

    for i in range(1, 6):
        camada = f'layer{i}'
        acuracias = dadosPorCamada[camada]['acc']
        f1_scores = dadosPorCamada[camada]['f1']
        dimensao = dadosPorCamada[camada]['dim']

        if len(acuracias) > 0:
            mediaAcc = np.mean(acuracias)
            desvioPadraoAcc = np.std(acuracias)
            mediaF1 = np.mean(f1_scores)
            resultadoFinal[camada] = {
                'accuracy_mean': mediaAcc,
                'accuracy_std': desvioPadraoAcc,
                'f1_score': mediaF1,
                'dim': dimensao
            }
            print(f"  {camada}: {mediaAcc:.2f}% (+/- {desvioPadraoAcc:.2f})")

    return resultadoFinal


def main():
    print("=" * 50)
    print("VGG-16 - Classificacao de Atributos de Roupas")
    print("Protocolo de Validacao: 70% treino / 30% teste")
    print("=" * 50)
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"usando gpu: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("sem gpu, vai demorar mais")

    modelo = criarExtrator(dispositivo=device)

    pastaDados = os.path.join(PASTA_RAIZ, 'data')
    pastaProtocolo = os.path.join(pastaDados, 'protocols')

    if not os.path.exists(pastaProtocolo):
        print(f"erro: pasta do protocolo nao encontrada: {pastaProtocolo}")
        return

    print("\n" + "=" * 50)
    print("CLASSIFICACAO POR COR")
    print("=" * 50)
    resultadoCor = rodarExperimentoComFolds(
        "COR",
        os.path.join(pastaProtocolo, 'folds_color_70_30', 'folds'),
        os.path.join(pastaDados, 'images', 'color'),
        modelo, device, tipoAtributo='color'
    )

    print("\n" + "=" * 50)
    print("CLASSIFICACAO POR TEXTURA")
    print("=" * 50)
    resultadoTextura = rodarExperimentoComFolds(
        "TEXTURA",
        os.path.join(pastaProtocolo, 'folds_texture_70_30', 'folds'),
        os.path.join(pastaDados, 'images', 'texture'),
        modelo, device, tipoAtributo='texture'
    )

    print("\nsalvando resultados...")
    for nomeArquivo, resultado in [('results_color_vgg16', resultadoCor), ('results_texture_vgg16', resultadoTextura)]:
        df = pd.DataFrame([{'camada': c, **d} for c, d in resultado.items()])
        caminhoArquivo = os.path.join(PASTA_RAIZ, f'{nomeArquivo}.csv')
        df.to_csv(caminhoArquivo, index=False)
        print(f"  {caminhoArquivo}")

    print(f"\nexperimento finalizado: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
