"""iBOT (ViT-S/16 auto-supervisionado).

Implementacao do ViT segue o repositorio oficial (https://github.com/bytedance/ibot).
Extrai features dos 12 blocos do transformer; nos experimentos do artigo usamos o
token CLS de cada bloco (extrairFeaturesComCLS).

Os pesos (.pth) nao vao no repositorio; ver checkpoints/README.md.
"""

import os
from functools import partial

import torch
import torch.nn as nn

PASTA_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASTA_CHECKPOINTS = os.path.join(PASTA_RAIZ, 'checkpoints')

PESOS = {
    'vit_small': os.path.join(PASTA_CHECKPOINTS, 'checkpoint_ibot_vits16.pth'),
    'vit_base':  os.path.join(PASTA_CHECKPOINTS, 'checkpoint_ibot_vitb16.pth'),
}


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mascara = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mascara.floor_()
    return x.div(keep_prob) * mascara


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Bloco padrao do ViT: atencao + MLP, ambos com conexao residual."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # LayerScale: so usado quando init_values > 0
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        y = self.attn(self.norm1(x))
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Divide a imagem em patches e projeta cada um em embed_dim."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformerIBot(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self.initWeights)

    def initWeights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepareTokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward(self, x):
        x = self.prepareTokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def vitSmall(patchSize=16, **kwargs):
    return VisionTransformerIBot(patch_size=patchSize, embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)


def vitBase(patchSize=16, **kwargs):
    return VisionTransformerIBot(patch_size=patchSize, embed_dim=768, depth=12,
                                 num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)


class ExtratorIBot:

    def __init__(self, modelo='vit_small', dispositivo='auto'):
        self.device = _resolver_dispositivo(dispositivo)
        self.tipoModelo = modelo
        self.numBlocos = 12
        self.dimEmbed = 384 if modelo == 'vit_small' else 768
        self.dimBlocos = {f'block{i}': self.dimEmbed for i in range(self.numBlocos)}
        self.features = {}
        self.hooks = []
        self.carregarModelo()

    def carregarModelo(self):
        self.model = vitSmall() if self.tipoModelo == 'vit_small' else vitBase()

        caminho = PESOS[self.tipoModelo]
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"pesos do iBOT nao encontrados em {caminho} (ver checkpoints/README.md)")

        checkpoint = torch.load(caminho, map_location='cpu', weights_only=False)
        pesos = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        # os checkpoints publicados vem com prefixo de wrapper
        pesos = {k.replace('backbone.', '').replace('module.', ''): v
                 for k, v in pesos.items()}
        self.model.load_state_dict(pesos, strict=False)

        self.model = self.model.to(self.device).eval()
        self.registrarHooks()
        print(f"[iBOT] ok | {self.tipoModelo} | device={self.device} | dim={self.dimEmbed}")

    def registrarHooks(self):
        def criarHook(indice):
            def hook(modulo, entrada, saida):
                self.features[f'block_{indice}'] = saida
            return hook

        for h in self.hooks:
            h.remove()
        self.hooks = []

        for i, bloco in enumerate(self.model.blocks[:self.numBlocos]):
            self.hooks.append(bloco.register_forward_hook(criarHook(i)))

    def _rodar(self, x):
        self.features = {}
        with torch.no_grad():
            self.model(x)

    def extrairFeatures(self, x, aplicarGAP=True):
        """Media dos tokens de patch (descarta o CLS)."""
        self._rodar(x)
        resultado = {}
        for i in range(self.numBlocos):
            feat = self.features.get(f'block_{i}')
            if feat is None:
                continue
            if aplicarGAP:
                if feat.dim() == 3:
                    feat = feat[:, 1:].mean(dim=1)
                elif feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
            resultado[f'block{i}'] = feat.cpu()
        return resultado

    def extrairFeaturesComCLS(self, x):
        """Token CLS de cada bloco — e o que os experimentos do artigo usam."""
        self._rodar(x)
        resultado = {}
        for i in range(self.numBlocos):
            feat = self.features.get(f'block_{i}')
            if feat is None:
                continue
            if feat.dim() == 3:
                feat = feat[:, 0]
            resultado[f'block{i}'] = feat.cpu()
        return resultado

    def pegarDimensoes(self):
        return dict(self.dimBlocos)


def _resolver_dispositivo(dispositivo):
    if dispositivo == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return dispositivo


def criarExtrator(modelo='vit_small', dispositivo='auto'):
    return ExtratorIBot(modelo=modelo, dispositivo=dispositivo)
