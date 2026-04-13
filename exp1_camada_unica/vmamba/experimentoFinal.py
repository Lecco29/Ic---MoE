#!/usr/bin/env python3
# experimento VMamba - classificacao de atributos visuais em roupas
# usando protocolo de validacao 70/30 com 5 folds
# modelo: VMamba (Visual Mamba - State Space Model)

import os
import sys
import torch
import pandas as pd
from datetime import datetime
from PIL import Image
from torchvision import transforms

# configura os caminhos do projeto pra importar os modulos
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, 'models'))
sys.path.insert(0, os.path.join(RAIZ, 'features'))
sys.path.insert(0, os.path.join(RAIZ, 'analysis'))

# imports dos modulos do projeto
from carregadorVMamba import criarExtrator
from extracaoFeatures import extrairFeatures
from avaliacaoKNN import avaliarKNNComFolds


def carregarImagensDeFolds(pastaProtocolo, tipoDataset, pastaImagens):
    """
    carrega todas as imagens baseado nos arquivos de fold
    junta treino e teste de todos os folds pra pegar todas as imagens unicas
    retorna as imagens transformadas, labels e nomes dos arquivos
    """
    
    # configura pasta de folds baseado no tipo de dataset
    if tipoDataset == 'color':
        pastaFolds = os.path.join(pastaProtocolo, 'folds_color_70_30', 'folds')
        prefixo = 'color'
    else:
        pastaFolds = os.path.join(pastaProtocolo, 'folds_texture_70_30', 'folds')
        prefixo = 'texture'
    
    # coleta todos os arquivos unicos de todos os folds
    # usa dicionario pra evitar duplicatas
    todosArquivos = {}  # nomeOriginal -> (classe, caminhoLocal)
    
    # le todos os folds (treino e teste)
    for numFold in range(1, 6):
        for tipo in ['train', 'test']:
            arquivoFold = os.path.join(pastaFolds, f'fold{numFold}-{tipo}.txt')
            
            with open(arquivoFold, 'r') as f:
                for linha in f:
                    partes = linha.strip().split(';')
                    if len(partes) >= 3:
                        nomeClasse = partes[1]
                        caminhoOriginal = partes[2]
                        nomeOriginal = os.path.basename(caminhoOriginal)
                        
                        # monta nome local no formato do projeto
                        nomeBase, extensao = os.path.splitext(nomeOriginal)
                        nomeLocal = f"{prefixo}_{nomeClasse}_{nomeBase}{extensao}"
                        caminhoLocal = os.path.join(pastaImagens, nomeLocal)
                        
                        # adiciona se ainda nao tiver
                        if nomeOriginal not in todosArquivos:
                            todosArquivos[nomeOriginal] = (nomeClasse, caminhoLocal, nomeOriginal)
    
    # transformacoes padrao do imagenet
    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # carrega as imagens
    listaImagens = []
    listaRotulos = []
    listaNomes = []
    
    for nomeOriginal, (classe, caminhoLocal, nomeArquivo) in todosArquivos.items():
        if os.path.exists(caminhoLocal):
            try:
                imagem = Image.open(caminhoLocal).convert('RGB')
                listaImagens.append(transformacao(imagem))
                listaRotulos.append(classe)
                listaNomes.append(nomeOriginal)  # guarda nome original pra usar nos folds
            except Exception as erro:
                print(f"erro ao carregar {caminhoLocal}: {erro}")
        else:
            print(f"arquivo nao encontrado: {caminhoLocal}")
    
    # verifica se carregou alguma imagem
    if len(listaImagens) == 0:
        raise RuntimeError(f"nenhuma imagem encontrada em {pastaImagens}")
    
    return torch.stack(listaImagens), listaRotulos, listaNomes


def rodarExperimento(nome, tipoDataset, pastaProtocolo, pastaImagens, extrator, dispositivo):
    """
    executa o experimento completo usando os folds pre-definidos
    extrai features de todas as imagens e avalia com knn em cada fold
    """
    
    print(f"\n{'='*50}")
    print(f"EXPERIMENTO: {nome}")
    print(f"{'='*50}")
    
    # carrega todas as imagens
    print("carregando imagens dos folds...")
    imagens, rotulos, nomes = carregarImagensDeFolds(pastaProtocolo, tipoDataset, pastaImagens)
    print(f"total: {len(imagens)} imagens, {len(set(rotulos))} classes")
    
    # extrai features de todas as imagens (faz uma vez so)
    print("extraindo features com vmamba...")
    features = extrairFeatures(extrator, imagens, dispositivo)
    
    # avalia com knn usando os folds pre-definidos
    print("avaliando com classificador knn...")
    resultados = avaliarKNNComFolds(features, rotulos, nomes, pastaProtocolo, tipoDataset)
    
    # mostra resultados de cada stage
    print(f"\nresultados {nome} (media dos 5 folds):")
    for estagio in ['stage1', 'stage2', 'stage3', 'stage4']:
        res = resultados[estagio]
        print(f"  {estagio}: {res['accuracy_mean']:.2f}% (+/- {res['accuracy_std']:.2f})")
    
    return resultados


def main():
    """
    funcao principal que executa todo o experimento
    testa o vmamba em classificacao de cor e textura
    """
    
    print("=" * 50)
    print("VMamba - Classificacao de Atributos de Roupas")
    print("Protocolo de Validacao: 70% treino / 30% teste")
    print("=" * 50)
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    # verifica se tem gpu disponivel pra acelerar
    if torch.cuda.is_available():
        dispositivo = 'cuda'
        print(f"usando gpu: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = 'cpu'
        print("gpu nao disponivel, usando cpu (vai demorar mais)")
    
    # carrega o modelo vmamba pre-treinado
    print("\ncarregando modelo vmamba...")
    extrator = criarExtrator(dispositivo=dispositivo)
    
    # configura as pastas de dados
    pastaData = os.path.join(RAIZ, 'data')
    pastaProtocolo = os.path.join(pastaData, 'protocols')
    
    # executa experimento de classificacao por cor
    resultadosCor = rodarExperimento(
        "COR",
        'color',
        pastaProtocolo,
        os.path.join(pastaData, 'images', 'color'),
        extrator, dispositivo
    )
    
    # executa experimento de classificacao por textura
    resultadosTextura = rodarExperimento(
        "TEXTURA",
        'texture',
        pastaProtocolo,
        os.path.join(pastaData, 'images', 'texture'),
        extrator, dispositivo
    )
    
    # salva os resultados em arquivos csv
    print("\nsalvando resultados em csv...")
    for nomeArquivo, resultado in [('color', resultadosCor), ('texture', resultadosTextura)]:
        df = pd.DataFrame([{'stage': estagio, **dados} for estagio, dados in resultado.items()])
        caminhoArquivo = os.path.join(RAIZ, f'results_{nomeArquivo}_final.csv')
        df.to_csv(caminhoArquivo, index=False)
        print(f"arquivo salvo: {caminhoArquivo}")
    
    print(f"\nexperimento finalizado: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
