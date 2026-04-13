# carregador da VGG-16 pre-treinada no ImageNet
# extrai features das 5 camadas de max pooling
# usa hooks do pytorch pra capturar as ativacoes intermediarias

import torch
import torch.nn as nn
from torchvision import models


class ExtratorVGG16:

    def __init__(self, dispositivo='auto'):
        if dispositivo == 'auto':
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo

        # dims depois do GAP em cada bloco
        self.dimCamadas = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512,
            'layer5': 512
        }

        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        print("[VGG-16] carregando modelo pre-treinado...")
        self.modelo = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo)
        self.modelo.eval()
        self.registrarHooks()
        print(f"[VGG-16] ok | device={self.dispositivo} | dims={self.dimCamadas}")

    def registrarHooks(self):
        # hooks nos max-poolings da vgg16 (indices 4, 9, 16, 23, 30)
        def criarHook(nomeCamada):
            def hook(modulo, entrada, saida):
                self.features[nomeCamada] = saida
            return hook

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        indicesPooling = {4: 'layer1', 9: 'layer2', 16: 'layer3', 23: 'layer4', 30: 'layer5'}
        for indice, nomeCamada in indicesPooling.items():
            h = self.modelo.features[indice].register_forward_hook(criarHook(nomeCamada))
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
    return ExtratorVGG16(dispositivo=dispositivo)
