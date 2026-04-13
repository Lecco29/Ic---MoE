# extracao de features do vmamba

import numpy as np
import torch
from tqdm import tqdm


def extrairFeatures(extrator, imagens, dispositivo, tamanhoLote=64):
    # extrai features de todas as imagens usando o vmamba
    # retorna dicionario com features de cada estagio
    
    todasFeatures = {f'stage{i+1}': [] for i in range(4)}
    numLotes = (len(imagens) + tamanhoLote - 1) // tamanhoLote
    
    for indiceLote in tqdm(range(numLotes), desc="Extraindo"):
        inicio = indiceLote * tamanhoLote
        fim = min((indiceLote + 1) * tamanhoLote, len(imagens))
        loteImagens = imagens[inicio:fim].to(dispositivo)
        
        # extrai features com global average pooling
        featuresLote = extrator.extrairFeatures(loteImagens, aplicarGAP=True)
        
        for estagio, feature in featuresLote.items():
            todasFeatures[estagio].append(feature.numpy())
    
    # junta todos os batches
    for estagio in todasFeatures:
        todasFeatures[estagio] = np.vstack(todasFeatures[estagio])
    
    return todasFeatures

