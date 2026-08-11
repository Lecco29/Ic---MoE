"""ResNet-50 pre-treinada no ImageNet.

Extrai features dos 4 blocos residuais (layer1..layer4), capturadas por
forward hooks e reduzidas a um vetor por imagem via GAP.
"""

import torch
from torchvision import models

# dimensoes de saida de cada bloco residual, depois do GAP
DIMS = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class ExtratorResNet50:

    def __init__(self, dispositivo='auto'):
        self.dispositivo = _resolver_dispositivo(dispositivo)
        self.dimCamadas = dict(DIMS)
        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        self.modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo).eval()
        self.registrarHooks()
        print(f"[ResNet-50] ok | device={self.dispositivo} | dims={self.dimCamadas}")

    def registrarHooks(self):
        def criarHook(nome):
            def hook(modulo, entrada, saida):
                self.features[nome] = saida
            return hook

        for h in self.hooks:
            h.remove()
        self.hooks = []

        for nome in DIMS:
            h = getattr(self.modelo, nome).register_forward_hook(criarHook(nome))
            self.hooks.append(h)

    def extrairFeatures(self, entrada, aplicarGAP=True):
        """Retorna {bloco: tensor [B, C]} para um batch [B, 3, H, W]."""
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
    return ExtratorResNet50(dispositivo=dispositivo)
