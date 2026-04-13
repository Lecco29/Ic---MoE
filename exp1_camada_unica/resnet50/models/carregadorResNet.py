# carregador da ResNet-50 pre-treinada no ImageNet
# extrai features dos 4 blocos residuais principais
# usa hooks do pytorch pra capturar as ativacoes intermediarias

import torch
import torch.nn as nn
from torchvision import models


class ExtratorResNet50:

    def __init__(self, dispositivo='auto'):
        if dispositivo == 'auto':
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo

        # dims de cada bloco residual depois do GAP
        self.dimCamadas = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }

        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        print("[ResNet-50] carregando modelo pre-treinado...")
        self.modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo)
        self.modelo.eval()
        self.registrarHooks()
        print(f"[ResNet-50] ok | device={self.dispositivo} | dims={self.dimCamadas}")

    def registrarHooks(self):
        def criarHook(nomeCamada):
            def hook(modulo, entrada, saida):
                self.features[nomeCamada] = saida
            return hook

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for nome in ['layer1', 'layer2', 'layer3', 'layer4']:
            h = getattr(self.modelo, nome).register_forward_hook(criarHook(nome))
            self.hooks.append(h)

    def extrairFeatures(self, entrada, aplicarGAP=True):
        self.features = {}
        with torch.no_grad():
            _ = self.modelo(entrada)

        resultado = {}
        for nomeCamada, feature in self.features.items():
            if aplicarGAP and len(feature.shape) == 4:
                feature = feature.mean(dim=[2, 3])
            resultado[nomeCamada] = feature.cpu()

        return resultado

    def pegarDimensoes(self):
        return self.dimCamadas.copy()


def criarExtrator(dispositivo='auto'):
    """funcao auxiliar pra criar o extrator de features"""
    return ExtratorResNet50(dispositivo=dispositivo)
