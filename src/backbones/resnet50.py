# carregador da ResNet-50 pre-treinada no ImageNet
# extrai features dos 4 blocos residuais principais
# usa hooks do pytorch pra capturar as ativacoes intermediarias

import torch
import torch.nn as nn
from torchvision import models


class ExtratorResNet50:
    """
    classe que carrega a ResNet-50 e extrai features de cada bloco residual
    a resnet50 tem 4 blocos residuais principais (layer1 ate layer4)
    cada bloco tem varios bottleneck blocks empilhados
    """

    def __init__(self, dispositivo='auto'):
        # configura o dispositivo (gpu ou cpu)
        if dispositivo == 'auto':
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo

        # dimensoes de saida de cada bloco residual (depois do gap)
        # esses valores sao fixos da arquitetura da resnet50
        self.dimCamadas = {
            'layer1': 256,   # primeiro bloco residual
            'layer2': 512,   # segundo bloco residual
            'layer3': 1024,  # terceiro bloco residual
            'layer4': 2048   # quarto bloco residual
        }

        # dicionario pra guardar as features capturadas pelos hooks
        self.features = {}
        self.hooks = []

        # carrega o modelo
        self.carregarModelo()

    def carregarModelo(self):
        """
        carrega a ResNet-50 com pesos pre-treinados no ImageNet
        registra os hooks nos blocos residuais
        """
        print("[ResNet-50] carregando modelo pre-treinado...")

        # carrega o modelo com pesos do imagenet
        self.modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo)
        self.modelo.eval()  # coloca em modo de avaliacao (desativa dropout etc)

        # registra os hooks pra capturar features intermediarias
        self.registrarHooks()

        print(f"[ResNet-50] modelo carregado com sucesso")
        print(f"[ResNet-50] pesos: ImageNet1K (pre-treinado)")
        print(f"[ResNet-50] dispositivo: {self.dispositivo}")
        print(f"[ResNet-50] blocos disponiveis: 4 (layer1, layer2, layer3, layer4)")
        print(f"[ResNet-50] dimensoes: {self.dimCamadas}")

    def registrarHooks(self):
        """
        registra forward hooks nos blocos residuais da ResNet-50
        a gente usa isso pra capturar as ativacoes intermediarias
        """

        def criarHook(nomeCamada):
            # funcao que cria o hook pra uma camada especifica
            def hook(modulo, entrada, saida):
                self.features[nomeCamada] = saida
            return hook

        # remove hooks antigos se tiver
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # mapeamento dos blocos residuais
        blocos = {
            'layer1': self.modelo.layer1,
            'layer2': self.modelo.layer2,
            'layer3': self.modelo.layer3,
            'layer4': self.modelo.layer4
        }

        # registra um hook em cada bloco
        for nomeCamada, bloco in blocos.items():
            hook = bloco.register_forward_hook(criarHook(nomeCamada))
            self.hooks.append(hook)

        print(f"[ResNet-50] {len(self.hooks)} hooks registrados")

    def extrairFeatures(self, entrada, aplicarGAP=True):
        """
        extrai features de todos os blocos pra um batch de imagens
        
        parametros:
            entrada: tensor de imagens [B, C, H, W]
            aplicarGAP: se True aplica global average pooling
        
        retorna:
            dicionario com features de cada bloco
        """
        
        # limpa as features anteriores
        self.features = {}

        # faz o forward pass (sem calcular gradientes)
        with torch.no_grad():
            _ = self.modelo(entrada)

        # processa as features capturadas
        resultado = {}
        for nomeCamada, feature in self.features.items():
            if aplicarGAP:
                # global average pooling: transforma [B, C, H, W] em [B, C]
                # faz a media espacial (nas dimensoes H e W)
                if len(feature.shape) == 4:
                    feature = feature.mean(dim=[2, 3])
            resultado[nomeCamada] = feature.cpu()

        return resultado

    def pegarDimensoes(self):
        """retorna as dimensoes de cada bloco"""
        return self.dimCamadas.copy()


def criarExtrator(dispositivo='auto'):
    """funcao auxiliar pra criar o extrator de features"""
    return ExtratorResNet50(dispositivo=dispositivo)
