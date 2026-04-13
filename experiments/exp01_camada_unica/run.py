#!/usr/bin/env python3
# experimento 1 - extrai features de cada camada do backbone e avalia com KNN
# protocolo: 5 folds, 70% treino 30% teste
# pra trocar o backbone e so mudar a variavel BACKBONE la embaixo

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from torchvision import transforms

# caminhos
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ   = os.path.dirname(os.path.dirname(PASTA_SCRIPT))
PASTA_EXP1   = os.path.join(PASTA_RAIZ, 'exp1_camada_unica')
PASTA_DADOS  = os.path.join(PASTA_RAIZ, 'data')

# troca aqui pra rodar com outro backbone: ibot | resnet50 | vgg16 | vmamba
BACKBONE = 'vgg16'

PASTA_BACKBONE = os.path.join(PASTA_EXP1, BACKBONE)
sys.path.insert(0, os.path.join(PASTA_BACKBONE, 'models'))
sys.path.insert(0, os.path.join(PASTA_BACKBONE, 'features'))
sys.path.insert(0, os.path.join(PASTA_BACKBONE, 'analysis'))

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
    """
    carrega as imagens de um arquivo de fold
    o arquivo tem o formato: indice;classe;caminho_original
    retorna as imagens ja transformadas e os labels
    """

    listaImagens = []
    listaLabels = []

    # le o arquivo do fold
    with open(arquivoFold, 'r') as f:
        linhas = f.readlines()

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        # separa as partes da linha (formato: indice;classe;caminho)
        partes = linha.split(';')
        if len(partes) < 3:
            continue

        classe = partes[1]
        caminhoOriginal = partes[2]

        # extrai informacoes do caminho original pra montar o nome local
        partesPath = caminhoOriginal.split('/')
        subpasta = partesPath[-2]  # nome da classe no caminho
        nomeArquivoOriginal = partesPath[-1]  # nome do arquivo

        # monta o nome no formato que ta salvo no projeto
        # formato: tipoAtributo_classe_nomeOriginal
        nomeArquivoLocal = f"{tipoAtributo}_{subpasta}_{nomeArquivoOriginal}"
        caminhoLocal = os.path.join(pastaBase, nomeArquivoLocal)

        # tenta carregar a imagem se existir
        if os.path.exists(caminhoLocal):
            try:
                img = Image.open(caminhoLocal).convert('RGB')
                listaImagens.append(TRANSFORMACAO(img))
                listaLabels.append(classe)
            except Exception as e:
                print(f"erro ao carregar imagem: {caminhoLocal}")

    # verifica se carregou alguma imagem
    if len(listaImagens) == 0:
        return None, None

    # retorna como tensor e lista de labels
    return torch.stack(listaImagens), listaLabels


def rodarExperimentoComFolds(nome, pastaFolds, pastaImagens, modelo, device, tipoAtributo):
    """
    executa o experimento usando os 5 folds pre-definidos
    pra cada fold extrai features e avalia com knn
    no final calcula a media e desvio padrao dos resultados
    """

    print(f"\n[{nome}]")

    # dicionario pra guardar os resultados de cada camada
    # a vgg16 tem 5 camadas de pooling que a gente extrai features
    dadosPorCamada = {f'layer{i}': {'acc': [], 'f1': [], 'dim': 0} for i in range(1, 6)}

    # processa cada um dos 5 folds
    for numFold in range(1, 6):
        print(f"\n  processando fold {numFold}/5:")

        # monta os caminhos dos arquivos de treino e teste do fold
        arquivoTreino = os.path.join(pastaFolds, f'fold{numFold}-train.txt')
        arquivoTeste = os.path.join(pastaFolds, f'fold{numFold}-test.txt')

        # verifica se os arquivos existem
        if not os.path.exists(arquivoTreino) or not os.path.exists(arquivoTeste):
            print(f"    arquivos do fold nao encontrados!")
            continue

        # carrega as imagens de treino
        print(f"    carregando imagens de treino...")
        imagensTreino, labelsTreino = carregarImagensDoFold(arquivoTreino, pastaImagens, tipoAtributo)
        if imagensTreino is None:
            print(f"    erro ao carregar imagens de treino")
            continue
        print(f"    treino: {len(imagensTreino)} imagens carregadas")

        # carrega as imagens de teste
        print(f"    carregando imagens de teste...")
        imagensTeste, labelsTeste = carregarImagensDoFold(arquivoTeste, pastaImagens, tipoAtributo)
        if imagensTeste is None:
            print(f"    erro ao carregar imagens de teste")
            continue
        print(f"    teste: {len(imagensTeste)} imagens carregadas")

        # extrai features das imagens de treino usando o modelo
        print(f"    extraindo features do conjunto de treino...")
        featuresTreino = extrairFeatures(modelo, imagensTreino, device)

        # extrai features das imagens de teste
        print(f"    extraindo features do conjunto de teste...")
        featuresTeste = extrairFeatures(modelo, imagensTeste, device)

        # avalia as features com knn (k=5)
        print(f"    avaliando com classificador knn...")
        resultados = avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste)

        # guarda os resultados de cada camada pro fold atual
        for camada, res in resultados.items():
            dadosPorCamada[camada]['acc'].append(res['accuracy'])
            dadosPorCamada[camada]['f1'].append(res['f1_score'])
            dadosPorCamada[camada]['dim'] = res['dim']

        # mostra qual foi a melhor camada nesse fold
        melhorCamada = max(resultados.keys(), key=lambda c: resultados[c]['accuracy'])
        print(f"    melhor camada: {melhorCamada} = {resultados[melhorCamada]['accuracy']:.2f}%")

    # calcula a media e desvio padrao dos 5 folds
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
    """
    funcao principal que executa todo o experimento
    testa a vgg16 em classificacao de cor e textura
    """
    
    print("=" * 50)
    print("VGG-16 - Classificacao de Atributos de Roupas")
    print("Protocolo de Validacao: 70% treino / 30% teste")
    print("=" * 50)
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")

    # verifica se tem gpu disponivel pra acelerar
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"usando gpu: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("gpu nao disponivel, usando cpu (vai demorar mais)")

    # carrega o modelo vgg16 pre-treinado
    print("\ncarregando modelo vgg-16...")
    modelo = criarExtrator(dispositivo=device)

    # configura as pastas de dados (centralizados em data/)
    pastaProtocolo = os.path.join(PASTA_DADOS, 'protocols')

    # verifica se a pasta do protocolo existe
    if not os.path.exists(pastaProtocolo):
        print(f"erro: pasta do protocolo nao encontrada: {pastaProtocolo}")
        return

    # executa experimento de classificacao por cor
    print("\n" + "=" * 50)
    print("EXPERIMENTO 1: CLASSIFICACAO POR COR")
    print("=" * 50)
    resultadoCor = rodarExperimentoComFolds(
        "COR",
        os.path.join(pastaProtocolo, 'folds_color_70_30', 'folds'),
        os.path.join(PASTA_DADOS, 'images', 'color'),
        modelo, device,
        tipoAtributo='color'
    )

    # executa experimento de classificacao por textura
    print("\n" + "=" * 50)
    print("EXPERIMENTO 2: CLASSIFICACAO POR TEXTURA")
    print("=" * 50)
    resultadoTextura = rodarExperimentoComFolds(
        "TEXTURA",
        os.path.join(pastaProtocolo, 'folds_texture_70_30', 'folds'),
        os.path.join(PASTA_DADOS, 'images', 'texture'),
        modelo, device,
        tipoAtributo='texture'
    )

    # salva os resultados em arquivos csv na pasta results/ do experimento
    print("\nsalvando resultados em csv...")
    pastaResultados = os.path.join(PASTA_SCRIPT, 'results')
    os.makedirs(pastaResultados, exist_ok=True)
    nomeBackbone = BACKBONE
    for nomeArquivo, resultado in [(f'{nomeBackbone}_color', resultadoCor), (f'{nomeBackbone}_texture', resultadoTextura)]:
        # converte pra dataframe do pandas
        df = pd.DataFrame([{'camada': c, **d} for c, d in resultado.items()])
        caminhoArquivo = os.path.join(pastaResultados, f'{nomeArquivo}.csv')
        df.to_csv(caminhoArquivo, index=False)
        print(f"arquivo salvo: {caminhoArquivo}")

    print(f"\nexperimento finalizado: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
