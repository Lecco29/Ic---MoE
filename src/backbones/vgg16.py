"""VGG-16 pre-treinada no ImageNet.

Extrai features das 5 camadas de max pooling (uma por bloco convolucional),
capturadas por forward hooks e reduzidas a um vetor por imagem via GAP.
"""

import torch
from torchvision import models

# dimensoes de saida de cada bloco, depois do GAP
DIMS = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'layer5': 512}

# indices dos max poolings dentro de model.features
INDICES_POOLING = {4: 'layer1', 9: 'layer2', 16: 'layer3', 23: 'layer4', 30: 'layer5'}


class ExtratorVGG16:

    def __init__(self, dispositivo='auto'):
        self.dispositivo = _resolver_dispositivo(dispositivo)
        self.dimCamadas = dict(DIMS)
        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        self.modelo = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo).eval()
        self.registrarHooks()
        print(f"[VGG-16] ok | device={self.dispositivo} | dims={self.dimCamadas}")

    def registrarHooks(self):
        def criarHook(nome):
            def hook(modulo, entrada, saida):
                self.features[nome] = saida
            return hook

        for h in self.hooks:
            h.remove()
        self.hooks = []

        for indice, nome in INDICES_POOLING.items():
            h = self.modelo.features[indice].register_forward_hook(criarHook(nome))
            self.hooks.append(h)

    def extrairFeatures(self, entrada, aplicarGAP=True):
        """Retorna {camada: tensor [B, C]} para um batch [B, 3, H, W]."""
        self.features = {}
        with torch.no_grad():
            self.modelo(entrada)

        resultado = {}
        for nome, feature in self.features.items():
            if aplicarGAP and feature.dim() == 4:
                feature = feature.mean(dim=[2, 3])  # media espacial H,W
            resultado[nome] = feature.cpu()
        return resultado

    def pegarDimensoes(self):
        return dict(self.dimCamadas)


def _resolver_dispositivo(dispositivo):
    if dispositivo == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dispositivo


def criarExtrator(dispositivo='auto'):
    return ExtratorVGG16(dispositivo=dispositivo)
