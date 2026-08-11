"""Modelo MoE para fusao adaptativa das camadas early/deep.

    backbone(x)  -> features da camada rasa (early) e da profunda (deep)
    hE, hD       -> projetam cada uma para uma dimensao comum d
    router       -> olha as duas e decide quanto usar de cada
    z            -> alpha * fE + beta * fD, com alpha + beta = 1

O backbone fica congelado; so hE, hD e o router treinam.

Nota sobre o iBOT: aqui o backbone e o ViT-S/16 supervisionado do timm, e nao o
checkpoint auto-supervisionado usado nos exp01/exp02. Foi assim que os numeros
publicados do MoE foram gerados; trocar o checkpoint muda os resultados.
"""

import os
import sys

import torch
import torch.nn as nn

PASTA_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
PASTA_VMAMBA_HF = os.path.join(PASTA_RAIZ, 'src', 'backbones', 'vmamba_hf')
PESOS_VMAMBA = os.path.join(PASTA_RAIZ, 'checkpoints', 'model.safetensors')


def gap(t):
    """Reduz [B, C, H, W] ou [B, N, C] a um vetor [B, C]."""
    if t.dim() == 4:
        return t.mean(dim=[2, 3])
    if t.dim() == 3:
        return t.mean(dim=1)
    return t


class CabecaProjecao(nn.Module):
    """Linear + LayerNorm + ReLU, levando features de dim qualquer para d."""

    def __init__(self, dim_entrada, dim_saida=256):
        super().__init__()
        self.linear = nn.Linear(dim_entrada, dim_saida)
        self.norm = nn.LayerNorm(dim_saida)
        self.ativacao = nn.ReLU()

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        return self.ativacao(self.norm(self.linear(x)))


class Router(nn.Module):
    """MLP de duas camadas que devolve os pesos alpha (early) e beta (deep).

    Codigo base fornecido pelo Prof. Alceu Britto Jr.
    """

    def __init__(self, d=256, oculto=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, oculto),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(oculto, 2),
        )

    def forward(self, fE, fD):
        pesos = torch.softmax(self.mlp(torch.cat([fE, fD], dim=1)), dim=1)
        alpha = pesos[:, 0:1]  # early / cor
        beta = pesos[:, 1:2]   # deep / textura
        return alpha, beta


class _BackboneComHooks(nn.Module):
    """Base dos backbones que capturam camadas intermediarias por hook."""

    def __init__(self):
        super().__init__()
        self._features = {}

    def _registrar(self, modulos, indices):
        for idx in indices:
            modulos[idx].register_forward_hook(self._hook(idx))

    def _hook(self, idx):
        def fn(modulo, entrada, saida):
            self._features[idx] = saida
        return fn

    def _congelar(self):
        for p in self.parameters():
            p.requires_grad = False


class BackboneIBOT(_BackboneComHooks):
    """ViT-S/16: early = bloco 2, deep = bloco 9, last = bloco 11."""

    IDX_EARLY, IDX_DEEP, IDX_LAST = 2, 9, 11
    DIM = 384

    def __init__(self, pre_treinado=True):
        super().__init__()
        import timm

        self.modelo = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k',
            pretrained=pre_treinado, num_classes=0)
        self._congelar()
        self._registrar(self.modelo.blocks, (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def forward(self, x):
        self._features = {}
        self.modelo(x)
        # token CLS (posicao 0) de cada bloco
        return tuple(self._features[i][:, 0, :]
                     for i in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def dimensoes(self):
        return self.DIM, self.DIM, self.DIM


class BackboneResNet50(nn.Module):
    """early = layer1 (256d), deep = layer3 (1024d), last = layer4 (2048d)."""

    def __init__(self, pre_treinado=True):
        super().__init__()
        from torchvision import models

        pesos = models.ResNet50_Weights.IMAGENET1K_V1 if pre_treinado else None
        resnet = models.resnet50(weights=pesos)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        fE = self.layer1(x)
        fD = self.layer3(self.layer2(fE))
        fG = self.layer4(fD)
        return fE, fD, fG

    def dimensoes(self):
        return 256, 1024, 2048


class BackboneVMamba(_BackboneComHooks):
    """early = layers[0] (192d), deep = layers[1] (384d), last = layers[2] (768d)."""

    IDX_EARLY, IDX_DEEP, IDX_LAST = 0, 1, 2

    def __init__(self, pre_treinado=True):
        super().__init__()
        sys.path.insert(0, PASTA_VMAMBA_HF)
        from configuration_vmamba import VMambaConfig
        from modeling_vmamba import VMambaForImageClassification

        self.modelo = VMambaForImageClassification(VMambaConfig())

        if pre_treinado:
            if not os.path.exists(PESOS_VMAMBA):
                raise FileNotFoundError(
                    f"pesos do VMamba nao encontrados em {PESOS_VMAMBA} "
                    f"(ver checkpoints/README.md)")
            from safetensors.torch import load_file
            self.modelo.load_state_dict(load_file(PESOS_VMAMBA))

        self._congelar()
        self._registrar(self.modelo.vmamba.layers,
                        (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def forward(self, x):
        self._features = {}
        self.modelo(x)
        return tuple(gap(self._features[i])
                     for i in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def dimensoes(self):
        return 192, 384, 768


class BackboneVGG16(_BackboneComHooks):
    """Max poolings: early = bloco 1 (64d), deep = bloco 3 (256d), last = bloco 5 (512d)."""

    IDX_EARLY, IDX_DEEP, IDX_LAST = 4, 16, 30

    def __init__(self, pre_treinado=True):
        super().__init__()
        from torchvision import models

        pesos = models.VGG16_Weights.IMAGENET1K_V1 if pre_treinado else None
        self.features = models.vgg16(weights=pesos).features
        self._congelar()
        self._registrar(self.features, (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def forward(self, x):
        self._features = {}
        self.features(x)
        return tuple(gap(self._features[i])
                     for i in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST))

    def dimensoes(self):
        return 64, 256, 512


class ModeloMoE(nn.Module):
    """Backbone congelado + projecoes hE/hD + router."""

    def __init__(self, backbone, d=256):
        super().__init__()
        self.backbone = backbone
        self.d = d

        dim_early, dim_deep, _ = backbone.dimensoes()
        self.hE = CabecaProjecao(dim_early, d)
        self.hD = CabecaProjecao(dim_deep, d)
        self.router = Router(d=d)

    def forward(self, x):
        with torch.no_grad():
            fE_bruto, fD_bruto, _ = self.backbone(x)

        fE = self.hE(fE_bruto)
        fD = self.hD(fD_bruto)
        alpha, beta = self.router(fE, fD)
        return alpha * fE + beta * fD, alpha, beta


BACKBONES_MOE = {
    'ibot': BackboneIBOT,
    'resnet50': BackboneResNet50,
    'vmamba': BackboneVMamba,
    'vgg16': BackboneVGG16,
}


def criar_modelo_moe(nome_backbone, d=256, pre_treinado=True):
    nome = nome_backbone.lower()
    if nome not in BACKBONES_MOE:
        raise ValueError(
            f"backbone '{nome_backbone}' nao reconhecido; "
            f"opcoes: {', '.join(BACKBONES_MOE)}")
    return ModeloMoE(BACKBONES_MOE[nome](pre_treinado), d=d)
