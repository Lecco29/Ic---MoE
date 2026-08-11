"""VMamba-Tiny (State Space Model).

Implementacao e pesos vem do port HuggingFace do VMamba (ver vmamba_hf/, MIT).
Extrai features dos 4 estagios via forward hooks, reduzidas por GAP.

Os pesos (model.safetensors) nao vao no repositorio; ver checkpoints/README.md.
"""

import os
import sys
import torch

PASTA_BACKBONES = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(os.path.dirname(PASTA_BACKBONES))
PESOS = os.path.join(PASTA_RAIZ, 'checkpoints', 'model.safetensors')

# o modelo do HF importa seus modulos por nome, entao a pasta precisa estar no path
sys.path.insert(0, os.path.join(PASTA_BACKBONES, 'vmamba_hf'))

# dimensoes observadas na saida de cada estagio (ja depois do downsample interno)
DIMS = {'stage1': 192, 'stage2': 384, 'stage3': 768, 'stage4': 768}


class ExtratorVMamba:

    def __init__(self, dispositivo='auto'):
        self.dispositivo = _resolver_dispositivo(dispositivo)
        self.dimEstagios = dict(DIMS)
        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        from modeling_vmamba import VMambaForImageClassification
        from configuration_vmamba import VMambaConfig
        from safetensors.torch import load_file

        self.modelo = VMambaForImageClassification(VMambaConfig())

        if not os.path.exists(PESOS):
            raise FileNotFoundError(
                f"pesos do VMamba nao encontrados em {PESOS} "
                f"(ver checkpoints/README.md)")
        self.modelo.load_state_dict(load_file(PESOS))

        self.modelo = self.modelo.to(self.dispositivo).eval()
        self.registrarHooks()
        print(f"[VMamba] ok | device={self.dispositivo} | dims={self.dimEstagios}")

    def registrarHooks(self):
        def criarHook(nome):
            def hook(modulo, entrada, saida):
                self.features[nome] = saida
            return hook

        for h in self.hooks:
            h.remove()
        self.hooks = []

        for indice, camada in enumerate(self.modelo.vmamba.layers):
            h = camada.register_forward_hook(criarHook(f'stage{indice + 1}'))
            self.hooks.append(h)

    def extrairFeatures(self, entrada, aplicarGAP=True):
        """Retorna {estagio: tensor [B, C]} para um batch [B, 3, H, W]."""
        self.features = {}
        with torch.no_grad():
            self.modelo(entrada)

        resultado = {}
        for nome, feature in self.features.items():
            if aplicarGAP:
                if feature.dim() == 4:
                    feature = feature.mean(dim=[2, 3])
                elif feature.dim() == 3:
                    feature = feature.mean(dim=1)
            resultado[nome] = feature.cpu()
        return resultado

    def pegarDimensoes(self):
        return dict(self.dimEstagios)


def _resolver_dispositivo(dispositivo):
    if dispositivo == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dispositivo


def criarExtrator(dispositivo='auto'):
    return ExtratorVMamba(dispositivo=dispositivo)
