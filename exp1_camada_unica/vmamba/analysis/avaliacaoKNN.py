# avaliacao com knn usando folds pre-definidos

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score


def carregarFold(caminhoArquivo, pastaImagensLocal):
    # carrega um arquivo de fold (train/val/test)
    # retorna lista de nomes de imagem e labels
    
    listaNomes = []
    listaRotulos = []
    
    with open(caminhoArquivo, 'r') as arquivoLeitura:
        for linha in arquivoLeitura:
            partes = linha.strip().split(';')
            if len(partes) >= 3:
                idClasse = int(partes[0])
                nomeClasse = partes[1]
                caminhoOriginal = partes[2]
                
                # extrai so o nome do arquivo
                nomeArquivo = os.path.basename(caminhoOriginal)
                
                listaNomes.append(nomeArquivo)
                listaRotulos.append(nomeClasse)
    
    return listaNomes, listaRotulos


def pegarIndices(nomesCompletos, nomesFold):
    # retorna os indices das imagens que estao no fold
    
    listaIndices = []
    for indice, nome in enumerate(nomesCompletos):
        nomeBase = os.path.basename(nome)
        if nomeBase in nomesFold:
            listaIndices.append(indice)
    
    return listaIndices


def avaliarKNNComFolds(features, rotulos, nomesImagens, pastaProtocolo, tipoDataset='color', k=5):
    # avalia as features com knn usando os folds pre-definidos
    # tipoDataset = 'color' ou 'texture'
    
    codificador = LabelEncoder()
    rotulosNumericos = codificador.fit_transform(rotulos)
    
    # caminho dos folds (protocolo 70/30)
    if tipoDataset == 'color':
        pastaFolds = os.path.join(pastaProtocolo, 'folds_color_70_30', 'folds')
    else:
        pastaFolds = os.path.join(pastaProtocolo, 'folds_texture_70_30', 'folds')
    
    resultados = {}
    
    for estagio, dadosFeatures in features.items():
        listaAcuracias = []
        listaF1Scores = []
        
        # processa cada fold (protocolo 70/30 - apenas train e test)
        for numFold in range(1, 6):
            # carrega arquivos do fold
            arquivoTreino = os.path.join(pastaFolds, f'fold{numFold}-train.txt')
            arquivoTeste = os.path.join(pastaFolds, f'fold{numFold}-test.txt')

            # carrega nomes de cada particao
            nomesTreino, _ = carregarFold(arquivoTreino, '')
            nomesTeste, _ = carregarFold(arquivoTeste, '')

            # protocolo 70/30 - apenas train e test
            nomesTreinoCompleto = set(nomesTreino)
            nomesTesteCompleto = set(nomesTeste)
            
            # encontra indices
            indicesTreino = []
            indicesTeste = []
            
            for indice, nome in enumerate(nomesImagens):
                nomeBase = os.path.basename(nome)
                if nomeBase in nomesTreinoCompleto:
                    indicesTreino.append(indice)
                elif nomeBase in nomesTesteCompleto:
                    indicesTeste.append(indice)
            
            if len(indicesTreino) == 0 or len(indicesTeste) == 0:
                print(f"  Aviso: fold {numFold} sem dados suficientes")
                continue
            
            # separa dados
            dadosTreino = dadosFeatures[indicesTreino]
            rotulosTreino = rotulosNumericos[indicesTreino]
            dadosTeste = dadosFeatures[indicesTeste]
            rotulosTeste = rotulosNumericos[indicesTeste]
            
            # treina e avalia knn
            classificadorKNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            classificadorKNN.fit(dadosTreino, rotulosTreino)
            predicoes = classificadorKNN.predict(dadosTeste)
            
            acuracia = accuracy_score(rotulosTeste, predicoes)
            f1Score = f1_score(rotulosTeste, predicoes, average='weighted')
            
            listaAcuracias.append(acuracia)
            listaF1Scores.append(f1Score)
        
        # calcula media e desvio
        resultados[estagio] = {
            'accuracy_mean': np.mean(listaAcuracias) * 100,
            'accuracy_std': np.std(listaAcuracias) * 100,
            'f1_score': np.mean(listaF1Scores) * 100,
            'dim': dadosFeatures.shape[1]
        }
    
    return resultados


# funcao antiga para compatibilidade (nao usa mais)
def avaliarKNN(features, rotulos, k=5, numFolds=5):
    # usa validacao cruzada aleatoria (metodo antigo)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    codificador = LabelEncoder()
    rotulosNumericos = codificador.fit_transform(rotulos)
    
    skf = StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=42)
    
    resultados = {}
    
    for estagio, dadosFeatures in features.items():
        classificadorKNN = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        acuracias = cross_val_score(classificadorKNN, dadosFeatures, rotulosNumericos, cv=skf, scoring='accuracy')
        
        classificadorKNN.fit(dadosFeatures, rotulosNumericos)
        predicoes = classificadorKNN.predict(dadosFeatures)
        f1Score = f1_score(rotulosNumericos, predicoes, average='weighted')
        
        resultados[estagio] = {
            'accuracy_mean': acuracias.mean() * 100,
            'accuracy_std': acuracias.std() * 100,
            'f1_score': f1Score * 100,
            'dim': dadosFeatures.shape[1]
        }
    
    return resultados
