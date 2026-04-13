# carregador da VGG-16 pre-treinada no ImageNet
# extrai features das 5 camadas de max pooling
# usa hooks do pytorch pra capturar as ativacoes intermediarias

import torch
import torch.nn as nn
from torchvision import models


class ExtratorVGG16:
    """
    classe que carrega a VGG-16 e extrai features de cada camada
    a vgg16 tem 5 blocos convolucionais, cada um seguido de max pooling
    a gente captura a saida de cada max pooling como features
    """

    def __init__(self, dispositivo='auto'):
        # configura o dispositivo (gpu ou cpu)
        if dispositivo == 'auto':
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo

        # dimensoes de saida de cada camada (depois do gap)
        # esses valores sao fixos da arquitetura da vgg16
        self.dimCamadas = {
            'layer1': 64,   # primeiro bloco
            'layer2': 128,  # segundo bloco
            'layer3': 256,  # terceiro bloco
            'layer4': 512,  # quarto bloco
            'layer5': 512   # quinto bloco
        }

        # dicionario pra guardar as features capturadas pelos hooks
        self.features = {}
        self.hooks = []

        # carrega o modelo
        self.carregarModelo()

    def carregarModelo(self):
        """
        carrega a VGG-16 com pesos pre-treinados no ImageNet
        registra os hooks nas camadas de max pooling
        """
        print("[VGG-16] carregando modelo pre-treinado...")

        # carrega o modelo com pesos do imagenet
        self.modelo = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.modelo = self.modelo.to(self.dispositivo)
        self.modelo.eval()  # coloca em modo de avaliacao (desativa dropout etc)

        # registra os hooks pra capturar features intermediarias
        self.registrarHooks()

        print(f"[VGG-16] modelo carregado com sucesso")
        print(f"[VGG-16] pesos: ImageNet1K (pre-treinado)")
        print(f"[VGG-16] dispositivo: {self.dispositivo}")
        print(f"[VGG-16] camadas disponiveis: 5 (uma por bloco)")
        print(f"[VGG-16] dimensoes: {self.dimCamadas}")

    def registrarHooks(self):
        """
        registra forward hooks nas camadas de max pooling da VGG-16
        hooks sao funcoes que sao chamadas quando a camada eh executada
        a gente usa isso pra capturar as ativacoes intermediarias
        
        na vgg16 as camadas de max pooling estao nos indices:
        4, 9, 16, 23, 30 do modulo features
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

        # mapeamento dos indices das camadas de max pooling
        # na arquitetura da vgg16
        indicesPooling = {
            4: 'layer1',   # depois do primeiro bloco (64 filtros)
            9: 'layer2',   # depois do segundo bloco (128 filtros)
            16: 'layer3',  # depois do terceiro bloco (256 filtros)
            23: 'layer4',  # depois do quarto bloco (512 filtros)
            30: 'layer5'   # depois do quinto bloco (512 filtros)
        }

        # registra um hook em cada camada de pooling
        for indice, nomeCamada in indicesPooling.items():
            camada = self.modelo.features[indice]
            hook = camada.register_forward_hook(criarHook(nomeCamada))
            self.hooks.append(hook)

        print(f"[VGG-16] {len(self.hooks)} hooks registrados")

    def extrairFeatures(self, entrada, aplicarGAP=True):
        """
        extrai features de todas as camadas pra um batch de imagens
        
        parametros:
            entrada: tensor de imagens [B, C, H, W]
            aplicarGAP: se True aplica global average pooling
        
        retorna:
            dicionario com features de cada camada
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
        """retorna as dimensoes de cada camada"""
        return self.dimCamadas.copy()


def criarExtrator(dispositivo='auto'):
    """funcao auxiliar pra criar o extrator de features"""
    return ExtratorVGG16(dispositivo=dispositivo)
