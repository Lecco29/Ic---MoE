# modelo MoE (Mixture of Experts) pra fusao adaptativa de features early/deep
#
# a ideia basica e:
#   backbone(x) -> features da camada rasa (early) e da camada profunda (deep)
#   hE projeta features early -> fE  [B, d]
#   hD projeta features deep  -> fD  [B, d]
#   router olha fE e fD e decide quanto usar de cada um
#   z = alpha * fE + beta * fD  (alpha + beta = 1)
#
# backbones disponiveis: ibot, resnet50, vmamba, vgg16

import torch
import torch.nn as nn


# cabeca de projecao: pega features de dimensao qualquer e joga pra dimensao d
# funciona com tensores [B, C] ou [B, C, H, W] (nesse caso faz GAP antes)
class CabecaProjecao(nn.Module):

    def __init__(self, dim_entrada, dim_saida=256):
        super().__init__()
        self.linear = nn.Linear(dim_entrada, dim_saida)
        self.norm = nn.LayerNorm(dim_saida)
        self.ativacao = nn.ReLU()

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # global average pooling
        return self.ativacao(self.norm(self.linear(x)))


# router: recebe fE e fD concatenados e aprende os pesos alpha e beta
# codigo base fornecido pelo Prof. Alceu Britto Jr.
class Router(nn.Module):

    def __init__(self, d=256, oculto=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, oculto),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(oculto, 2),
        )

    def forward(self, fE, fD):
        g = torch.cat([fE, fD], dim=1)   # junta os dois vetores
        logits = self.mlp(g)
        pesos = torch.softmax(logits, dim=1)
        alpha = pesos[:, 0:1]  # peso pra early (cor)
        beta = pesos[:, 1:2]   # peso pra deep (textura)
        return alpha, beta


# backbone do ibot - usa ViT-S/16 do timm com hooks nos blocos
# early = bloco 2, deep = bloco 9, last = bloco 11
class BackboneIBOT(nn.Module):

    IDX_EARLY = 2
    IDX_DEEP = 9
    IDX_LAST = 11
    DIM = 384  # dimensao do ViT-S

    def __init__(self, pre_treinado=True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("precisa do timm: pip install timm")

        self.modelo = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k',
            pretrained=pre_treinado,
            num_classes=0,
        )
        # congela o backbone, so o router e as cabecas vao treinar
        for p in self.modelo.parameters():
            p.requires_grad = False

        self._features = {}
        for idx in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST):
            self.modelo.blocks[idx].register_forward_hook(self._hook(idx))

    def _hook(self, idx):
        def fn(modulo, entrada, saida):
            self._features[idx] = saida
        return fn

    def forward(self, x):
        self._features = {}
        _ = self.modelo(x)
        # pega o CLS token (posicao 0) de cada bloco
        fE = self._features[self.IDX_EARLY][:, 0, :]
        fD = self._features[self.IDX_DEEP][:, 0, :]
        fG = self._features[self.IDX_LAST][:, 0, :]
        return fE, fD, fG

    def dimensoes(self):
        return self.DIM, self.DIM, self.DIM


# backbone da resnet50
# early = layer1 (256d), deep = layer3 (1024d), last = layer4 (2048d)
class BackboneResNet50(nn.Module):

    def __init__(self, pre_treinado=True):
        super().__init__()
        from torchvision import models
        pesos = models.ResNet50_Weights.IMAGENET1K_V1 if pre_treinado else None
        resnet = models.resnet50(weights=pesos)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 256 canais
        self.layer2 = resnet.layer2   # 512 canais
        self.layer3 = resnet.layer3   # 1024 canais
        self.layer4 = resnet.layer4   # 2048 canais

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        fE = self.layer1(x)       # early
        x = self.layer2(fE)
        fD = self.layer3(x)       # deep
        fG = self.layer4(fD)      # last
        return fE, fD, fG

    def dimensoes(self):
        return 256, 1024, 2048


# backbone do vmamba
# as dimensoes reais dos layers foram verificadas empiricamente:
# layers[0]=192d, layers[1]=384d, layers[2]=768d
class BackboneVMamba(nn.Module):

    IDX_EARLY = 0
    IDX_DEEP = 1
    IDX_LAST = 2

    def __init__(self, pre_treinado=True):
        super().__init__()
        import os, sys

        # sobe 4 niveis de pasta pra chegar na raiz do projeto
        raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        pasta_vmamba = os.path.join(raiz, 'exp1_camada_unica', 'vmamba')
        sys.path.insert(0, pasta_vmamba)

        from modeling_vmamba import VMambaForImageClassification
        from configuration_vmamba import VMambaConfig

        config = VMambaConfig()
        self.modelo = VMambaForImageClassification(config)

        if pre_treinado:
            caminho_pesos = os.path.join(pasta_vmamba, 'model.safetensors')
            if os.path.exists(caminho_pesos):
                from safetensors.torch import load_file
                estado = load_file(caminho_pesos)
                self.modelo.load_state_dict(estado)
                print("[VMamba] pesos carregados!")
            else:
                print("[VMamba] pesos nao encontrados, usando aleatorios")

        for p in self.modelo.parameters():
            p.requires_grad = False

        self._features = {}
        vmamba = self.modelo.vmamba
        for idx in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST):
            vmamba.layers[idx].register_forward_hook(self._hook(idx))

    def _hook(self, idx):
        def fn(modulo, entrada, saida):
            self._features[idx] = saida
        return fn

    def forward(self, x):
        self._features = {}
        _ = self.modelo(x)

        def gap(t):
            if t.dim() == 4:
                return t.mean(dim=[2, 3])
            elif t.dim() == 3:
                return t.mean(dim=1)
            return t

        fE = gap(self._features[self.IDX_EARLY])
        fD = gap(self._features[self.IDX_DEEP])
        fG = gap(self._features[self.IDX_LAST])
        return fE, fD, fG

    def dimensoes(self):
        return 192, 384, 768


# backbone da vgg16
# early = maxpool do bloco 1 (idx=4, 64d)
# deep  = maxpool do bloco 3 (idx=16, 256d)
# last  = maxpool do bloco 5 (idx=30, 512d)
class BackboneVGG16(nn.Module):

    IDX_EARLY = 4
    IDX_DEEP = 16
    IDX_LAST = 30

    def __init__(self, pre_treinado=True):
        super().__init__()
        from torchvision import models
        pesos = models.VGG16_Weights.IMAGENET1K_V1 if pre_treinado else None
        vgg = models.vgg16(weights=pesos)
        self.features = vgg.features

        for p in self.parameters():
            p.requires_grad = False

        self._features = {}
        for idx in (self.IDX_EARLY, self.IDX_DEEP, self.IDX_LAST):
            self.features[idx].register_forward_hook(self._hook(idx))

    def _hook(self, idx):
        def fn(modulo, entrada, saida):
            self._features[idx] = saida
        return fn

    def forward(self, x):
        self._features = {}
        _ = self.features(x)

        def gap(t):
            return t.mean(dim=[2, 3]) if t.dim() == 4 else t

        fE = gap(self._features[self.IDX_EARLY])
        fD = gap(self._features[self.IDX_DEEP])
        fG = gap(self._features[self.IDX_LAST])
        return fE, fD, fG

    def dimensoes(self):
        return 64, 256, 512


# modelo completo: backbone + cabecas de projecao + router
class ModeloMoE(nn.Module):

    def __init__(self, backbone, d=256):
        super().__init__()
        self.backbone = backbone
        dim_early, dim_deep, _ = backbone.dimensoes()

        self.hE = CabecaProjecao(dim_early, d)
        self.hD = CabecaProjecao(dim_deep, d)
        self.router = Router(d=d)
        self.d = d

    def forward(self, x):
        with torch.no_grad():
            fE_raw, fD_raw, _ = self.backbone(x)

        fE = self.hE(fE_raw)
        fD = self.hD(fD_raw)
        alpha, beta = self.router(fE, fD)
        z = alpha * fE + beta * fD  # vetor final ponderado
        return z, alpha, beta


# funcao auxiliar pra criar o modelo pelo nome do backbone
def criar_modelo_moe(nome_backbone, d=256, pre_treinado=True):
    nome = nome_backbone.lower()
    if nome == 'ibot':
        backbone = BackboneIBOT(pre_treinado)
    elif nome == 'resnet50':
        backbone = BackboneResNet50(pre_treinado)
    elif nome == 'vmamba':
        backbone = BackboneVMamba(pre_treinado)
    elif nome == 'vgg16':
        backbone = BackboneVGG16(pre_treinado)
    else:
        raise ValueError(f"backbone '{nome_backbone}' nao reconhecido. opcoes: ibot, resnet50, vmamba, vgg16")
    return ModeloMoE(backbone, d=d)
