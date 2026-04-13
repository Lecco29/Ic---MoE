# extracao de features da VGG-16 em batches

import numpy as np
import torch
from tqdm import tqdm


def extrairFeatures(extrator, imagens, dispositivo, tamanhoBatch=64):
    # 5 camadas de pooling na vgg16
    todasFeatures = {f'layer{i}': [] for i in range(1, 6)}
    numBatches = (len(imagens) + tamanhoBatch - 1) // tamanhoBatch

    for indiceBatch in tqdm(range(numBatches), desc="extraindo features"):
        inicio = indiceBatch * tamanhoBatch
        fim = min((indiceBatch + 1) * tamanhoBatch, len(imagens))
        batchImagens = imagens[inicio:fim].to(dispositivo)
        featuresBatch = extrator.extrairFeatures(batchImagens, aplicarGAP=True)
        for nomeCamada, featCamada in featuresBatch.items():
            todasFeatures[nomeCamada].append(featCamada.numpy())

    for nomeCamada in todasFeatures:
        todasFeatures[nomeCamada] = np.vstack(todasFeatures[nomeCamada])

    return todasFeatures
