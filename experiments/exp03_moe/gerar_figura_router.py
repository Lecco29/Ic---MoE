#!/usr/bin/env python3
"""
Gera a figura do comportamento do roteador MoE (distribuição de α e β).
Treina o MoE para 'color' e para 'texture' em um fold e coleta os
pesos α produzidos pelo router sobre o conjunto de teste.

Saída: ICMLA/fig_router_behavior.pdf  e  ICMLA/fig_router_behavior.png

Uso:
  python gerar_figura_router.py
  python gerar_figura_router.py --backbone resnet50 --epocas 25 --fold 1
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ   = os.path.dirname(os.path.dirname(PASTA_SCRIPT))
PASTA_DADOS  = os.path.join(PASTA_RAIZ, 'data')
PASTA_ICMLA  = os.path.join(PASTA_RAIZ, 'ICMLA')

sys.path.insert(0, PASTA_SCRIPT)
from models.moe import criar_modelo_moe


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ConjuntoDados(TorchDataset):

    def __init__(self, caminhos, rotulos, transform=None):
        self.caminhos  = caminhos
        self.rotulos   = rotulos
        self.transform = transform
        classes = sorted(set(rotulos))
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        img = Image.open(self.caminhos[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.c2i[self.rotulos[idx]], dtype=torch.long)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------
def carregar_fold(arquivo, pasta_imagens, atributo):
    caminhos, rotulos = [], []
    with open(arquivo) as f:
        for linha in f:
            partes = linha.strip().split(';')
            if len(partes) < 3:
                continue
            rotulo = partes[1]
            original = partes[2]
            nome = f"{atributo}_{original.split('/')[-2]}_{original.split('/')[-1]}"
            caminho = os.path.join(pasta_imagens, nome)
            if os.path.exists(caminho):
                caminhos.append(caminho)
                rotulos.append(rotulo)
    return caminhos, rotulos


def treinar_moe(modelo, loader_treino, num_classes, epocas, dispositivo,
                lr=1e-4, lambda_l2_logit=1.0):
    """
    Treina o MoE com regularização L2 direta nos logits do roteador.

    Regularizar a saída do softmax (alpha, beta) não funciona quando o roteador
    colapsou: o gradiente de entropia vai a zero devido à derivada do softmax
    (d_alpha/d_logit = alpha*(1-alpha) → 0 quando alpha → 0 ou 1).

    A solução é regularizar os LOGITS (antes do softmax) via hook na última
    camada linear do router.  A penalidade L2 nos logits mantém-nos próximos de
    zero, preservando alguma variabilidade no roteamento.  O gradiente da CE
    ainda empurra logit[0] para cima em tarefas de COR (camada rasa discrimina
    mais) e logit[1] para cima em tarefas de TEXTURA (camada profunda discrimina
    mais), produzindo distribuições de α distinguíveis entre as duas tarefas.
    """
    cabeca = nn.Linear(modelo.d, num_classes).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    params = (list(modelo.hE.parameters()) +
              list(modelo.hD.parameters()) +
              list(modelo.router.parameters()) +
              list(cabeca.parameters()))
    opt   = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)

    # hook para capturar os logits (saída da última Linear do MLP do router)
    _captured_logits = []
    def _hook(module, inp, out):
        _captured_logits.clear()
        _captured_logits.append(out)

    hook_handle = modelo.router.mlp[-1].register_forward_hook(_hook)

    modelo.train()
    cabeca.train()

    for epoca in range(epocas):
        loss_ep, l2_ep = 0.0, 0.0
        n = 0
        for imgs, rots in loader_treino:
            imgs, rots = imgs.to(dispositivo), rots.to(dispositivo)
            opt.zero_grad()
            z, alpha, beta = modelo(imgs)

            loss_ce  = criterio(cabeca(z), rots)
            logits   = _captured_logits[0]           # [B, 2] — antes do softmax
            loss_l2  = lambda_l2_logit * logits.pow(2).mean()

            loss = loss_ce + loss_l2
            loss.backward()
            opt.step()

            loss_ep += loss_ce.item() * rots.size(0)
            l2_ep   += loss_l2.item() * rots.size(0)
            n += rots.size(0)
        sched.step()
        if (epoca + 1) % 5 == 0 or epoca == 0:
            print(f"    época {epoca+1:3d}/{epocas}  "
                  f"ce={loss_ep/n:.4f}  l2={l2_ep/n:.4f}")

    hook_handle.remove()
    return modelo


@torch.no_grad()
def coletar_alphas(modelo, loader, dispositivo):
    """Retorna array numpy com α de cada imagem do loader."""
    modelo.eval()
    alphas = []
    for imgs, _ in loader:
        imgs = imgs.to(dispositivo)
        _, alpha, _ = modelo(imgs)          # alpha: [B, 1]
        alphas.append(alpha.cpu().numpy().flatten())
    return np.concatenate(alphas)


# ------------------------------------------------------------------
# Figura
# ------------------------------------------------------------------
def gerar_figura(alphas_color, alphas_texture, backbone, fold, pasta_saida):
    """
    Violin + strip plot mostrando a distribuição de α para cada tarefa.
    α alto  → roteador prefere camada rasa (early = cor)
    α baixo → roteador prefere camada profunda (deep = textura)
    """
    os.makedirs(pasta_saida, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    data    = [alphas_color, alphas_texture]
    labels  = ['Color\n(color-trained)', 'Texture\n(texture-trained)']
    palette = ['#1976D2', '#E65100']   # azul escuro, laranja escuro

    parts = ax.violinplot(data, positions=[1, 2],
                          showmeans=True, showmedians=False,
                          showextrema=True)

    for pc, color in zip(parts['bodies'], palette):
        pc.set_facecolor(color)
        pc.set_edgecolor('none')
        pc.set_alpha(0.65)

    for key in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
        if key in parts:
            parts[key].set_color('#333333')
            parts[key].set_linewidth(1.3)

    # strip de pontos individuais
    rng = np.random.default_rng(seed=42)
    for i, (d, color) in enumerate(zip(data, palette)):
        jitter = rng.normal(0, 0.035, size=len(d))
        ax.scatter(i + 1 + jitter, d, s=6, alpha=0.35,
                   color=color, zorder=2, linewidths=0)

    ax.axhline(0.5, color='#888888', linestyle='--',
               linewidth=0.9, alpha=0.7, label='α = 0.5 (equal weight)')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Router weight  α  (early layer)', fontsize=10)
    ax.set_ylim(-0.02, 1.02)

    p1 = mpatches.Patch(facecolor='#1976D2', alpha=0.7,
                         label=f'Color model  (mean α={alphas_color.mean():.2f})')
    p2 = mpatches.Patch(facecolor='#E65100', alpha=0.7,
                         label=f'Texture model  (mean α={alphas_texture.mean():.2f})')
    ax.legend(handles=[p1, p2], fontsize=8, loc='lower right',
              framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'Router Behavior — {backbone.upper()}  (fold {fold})',
                 fontsize=10)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        caminho = os.path.join(pasta_saida, f'fig_router_behavior.{ext}')
        plt.savefig(caminho, dpi=200, bbox_inches='tight')
        print(f"  Salvo: {caminho}")

    plt.close()

    # estatísticas no console
    print(f"\n  α  color   : μ={alphas_color.mean():.3f}  σ={alphas_color.std():.3f}"
          f"  mediana={np.median(alphas_color):.3f}")
    print(f"  α  texture : μ={alphas_texture.mean():.3f}  σ={alphas_texture.std():.3f}"
          f"  mediana={np.median(alphas_texture):.3f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Gera figura do comportamento do roteador MoE')
    parser.add_argument('--backbone', default='ibot',
                        choices=['ibot', 'resnet50', 'vgg16', 'vmamba'])
    parser.add_argument('--epocas', type=int, default=30,
                        help='épocas de treino por tarefa (padrão: 30)')
    parser.add_argument('--fold', type=int, default=1,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument('--lambda_l2', type=float, default=0.3,
                        help='penalidade L2 nos logits do router (padrão: 0.3)')
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo : {dispositivo}")
    print(f"Backbone    : {args.backbone}")
    print(f"Épocas      : {args.epocas}")
    print(f"Fold        : {args.fold}")

    alphas_por_atributo = {}

    for atributo in ['color', 'texture']:
        print(f"\n{'='*55}")
        print(f"  Treinando MoE → {atributo.upper()} (fold {args.fold})")
        print(f"{'='*55}")

        pasta_img   = os.path.join(PASTA_DADOS, 'images', atributo)
        pasta_folds = os.path.join(PASTA_DADOS, 'protocols',
                                   f'folds_{atributo}_70_30', 'folds')

        arq_tr = os.path.join(pasta_folds, f'fold{args.fold}-train.txt')
        arq_te = os.path.join(pasta_folds, f'fold{args.fold}-test.txt')

        cam_tr, rot_tr = carregar_fold(arq_tr, pasta_img, atributo)
        cam_te, rot_te = carregar_fold(arq_te, pasta_img, atributo)
        print(f"  treino: {len(cam_tr)} imgs  |  teste: {len(cam_te)} imgs")

        ds_tr = ConjuntoDados(cam_tr, rot_tr, get_transforms(train=True))
        ds_te = ConjuntoDados(cam_te, rot_te, get_transforms(train=False))

        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True,
                               num_workers=4, pin_memory=True)
        loader_te = DataLoader(ds_te, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)

        modelo = criar_modelo_moe(args.backbone, d=256,
                                  pre_treinado=True).to(dispositivo)

        print(f"  Treinando {args.epocas} épocas (λ_l2={args.lambda_l2})...")
        modelo = treinar_moe(modelo, loader_tr, ds_tr.num_classes,
                             args.epocas, dispositivo,
                             lambda_l2_logit=args.lambda_l2)

        print(f"  Coletando pesos α do conjunto de teste...")
        alphas = coletar_alphas(modelo, loader_te, dispositivo)
        alphas_por_atributo[atributo] = alphas
        print(f"  α médio: {alphas.mean():.3f} ± {alphas.std():.3f}")

    print(f"\n{'='*55}")
    print("  Gerando figura...")
    print(f"{'='*55}")
    gerar_figura(
        alphas_color   = alphas_por_atributo['color'],
        alphas_texture = alphas_por_atributo['texture'],
        backbone       = args.backbone,
        fold           = args.fold,
        pasta_saida    = PASTA_ICMLA,
    )
    print("\nFigura gerada com sucesso!")


if __name__ == '__main__':
    main()
