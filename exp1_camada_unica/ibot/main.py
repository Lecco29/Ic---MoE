#!/usr/bin/env python3
# experimento iBOT - classificacao de atributos visuais em roupas
# usando protocolo de validacao 70/30 com 5 folds
# modelo: iBOT (Image BERT Pre-Training with Online Tokenizer)

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

from carregadorIBot import criarExtrator
from extracaoFeatures import extrairFeaturesComCLS
from avaliacaoKNN import avaliarKNNComFolds


# transformacoes padrao pra imagens do imagenet
# o ibot usa o mesmo padrao de normalizacao do imagenet
TRANSFORMACAO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


def carregarImagensDoFold(arquivoFold, pastaBase, tipoAtributo):
    # formato: indice;classe;caminho_original
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

        # arquivos locais seguem: tipoAtributo_classe_nomeOriginal
        partesPath = caminhoOriginal.split('/')
        nomeArquivoLocal = f"{tipoAtributo}_{partesPath[-2]}_{partesPath[-1]}"
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

    # vit-small tem 12 blocos, todos com dim 384
    dadosPorBloco = {f'block{i}': {'acc': [], 'f1': [], 'dim': 384} for i in range(12)}

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

        # usa CLS token (nao a media dos patches)
        featuresTreino = extrairFeaturesComCLS(modelo, imagensTreino, device)
        featuresTeste = extrairFeaturesComCLS(modelo, imagensTeste, device)
        resultados = avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste)

        for bloco, res in resultados.items():
            dadosPorBloco[bloco]['acc'].append(res['accuracy'])
            dadosPorBloco[bloco]['f1'].append(res['f1_score'])
            dadosPorBloco[bloco]['dim'] = res['dim']

        melhorBloco = max(resultados.keys(), key=lambda b: resultados[b]['accuracy'])
        print(f"    melhor: {melhorBloco} = {resultados[melhorBloco]['accuracy']:.2f}%")

    print(f"\nresultados finais {nome} (media dos 5 folds):")
    resultadoFinal = {}

    for i in range(12):
        bloco = f'block{i}'
        acuracias = dadosPorBloco[bloco]['acc']
        if len(acuracias) > 0:
            mediaAcc = np.mean(acuracias)
            desvioPadraoAcc = np.std(acuracias)
            mediaF1 = np.mean(dadosPorBloco[bloco]['f1'])
            resultadoFinal[bloco] = {
                'accuracy_mean': mediaAcc,
                'accuracy_std': desvioPadraoAcc,
                'f1_score': mediaF1,
                'dim': dadosPorBloco[bloco]['dim']
            }
            print(f"  {bloco}: {mediaAcc:.2f}% (+/- {desvioPadraoAcc:.2f})")

    return resultadoFinal


def main():
    print("=" * 50)
    print("iBOT - Classificacao de Atributos de Roupas")
    print("Protocolo de Validacao: 70% treino / 30% teste")
    print("=" * 50)
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"usando gpu: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("sem gpu, vai demorar bastante")

    modelo = criarExtrator(modelo='vit_small', dispositivo=device)

    pastaDados = os.path.join(PASTA_RAIZ, 'data')
    pastaProtocolo = os.path.join(pastaDados, 'protocols')

    if not os.path.exists(pastaProtocolo):
        print(f"erro: protocolo nao encontrado em {pastaProtocolo}")
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
    for nomeArquivo, resultado in [('results_color_ibot', resultadoCor), ('results_texture_ibot', resultadoTextura)]:
        df = pd.DataFrame([{'bloco': b, **d} for b, d in resultado.items()])
        caminhoArquivo = os.path.join(PASTA_RAIZ, f'{nomeArquivo}.csv')
        df.to_csv(caminhoArquivo, index=False)
        print(f"  {caminhoArquivo}")
    
    print(f"\nexperimento finalizado: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
