#!/usr/bin/env python3
"""Figura do comportamento do router MoE (distribuicao dos pesos alpha).

Treina o MoE em cor e em textura no mesmo fold e coleta os alpha que o router
produz sobre o conjunto de teste. O modelo treinado em cor pende para a camada
rasa e o treinado em textura para a profunda — e o que a figura mostra.

Gera a Figura 5 do artigo, em outputs/figures/.

Uso:
  python experiments/exp03_moe/gerar_figura_router.py
  python experiments/exp03_moe/gerar_figura_router.py --backbone resnet50 --epocas 25
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(os.path.dirname(PASTA_SCRIPT))
sys.path.insert(0, PASTA_SCRIPT)
sys.path.insert(0, PASTA_RAIZ)

from models.moe import BACKBONES_MOE, criar_modelo_moe
from run import ConjuntoDados, pegar_transformacoes
from src.dados import listar_fold

PASTA_FIGURAS = os.path.join(PASTA_RAIZ, 'outputs', 'figures')


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
        _, alpha, _ = modelo(imgs)
        alphas.append(alpha.cpu().numpy().flatten())
    return np.concatenate(alphas)


@torch.no_grad()
def coletar_alphas_com_classes(modelo, loader, dispositivo, idx_para_classe):
    """
    Retorna (alphas, classes_str) — arrays paralelos com α e nome da classe
    de cada imagem do loader.
    idx_para_classe: dict {idx_int: nome_str}
    """
    modelo.eval()
    alphas, classes = [], []
    for imgs, rots in loader:
        imgs = imgs.to(dispositivo)
        _, alpha, _ = modelo(imgs)
        alphas.append(alpha.cpu().numpy().flatten())
        classes.extend([idx_para_classe[int(r)] for r in rots.cpu()])
    return np.concatenate(alphas), classes


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


def gerar_figura_por_classe(
    alphas_color, classes_color,
    alphas_texture, classes_texture,
    backbone, fold, pasta_saida
):
    """
    Boxplot de α por classe para o modelo treinado em color e em texture.

    Dois subplots lado a lado:
      - Esquerda: modelo treinado em COLOR (esperado: α alto em todas as classes)
      - Direita:  modelo treinado em TEXTURE (esperado: α baixo em todas as classes)

    Permite verificar se o router é consistente entre classes dentro de cada tarefa.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    configs = [
        (alphas_color,   classes_color,   'Color-trained model',   '#1976D2'),
        (alphas_texture, classes_texture, 'Texture-trained model', '#E65100'),
    ]

    for ax, (alphas, classes, titulo, cor) in zip(axes, configs):
        classes_unicas = sorted(set(classes))
        dados_por_classe = [
            [alphas[i] for i, c in enumerate(classes) if c == cls]
            for cls in classes_unicas
        ]

        bp = ax.boxplot(dados_por_classe, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker='o', markersize=2, alpha=0.4))

        for patch in bp['boxes']:
            patch.set_facecolor(cor)
            patch.set_alpha(0.6)

        ax.axhline(0.5, color='#888888', linestyle='--', linewidth=0.9, alpha=0.7,
                   label='α = 0.5')
        ax.set_xticks(range(1, len(classes_unicas) + 1))
        ax.set_xticklabels(classes_unicas, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Router weight  α  (early layer)', fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(titulo, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # anota média por classe
        for i, (cls, d) in enumerate(zip(classes_unicas, dados_por_classe)):
            ax.text(i + 1, max(d) + 0.03, f'{np.mean(d):.2f}',
                    ha='center', va='bottom', fontsize=6, color='#333333')

    fig.suptitle(
        f'Router per-class — {backbone.upper()}  (fold {fold})',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        caminho = os.path.join(pasta_saida, f'fig_router_por_classe.{ext}')
        plt.savefig(caminho, dpi=200, bbox_inches='tight')
        print(f"  Salvo: {caminho}")
    plt.close()

    # estatísticas por classe
    print("\n  α por classe — modelo COLOR:")
    for cls in sorted(set(classes_color)):
        vals = [alphas_color[i] for i, c in enumerate(classes_color) if c == cls]
        print(f"    {cls:<12}: μ={np.mean(vals):.3f}  σ={np.std(vals):.3f}")
    print("\n  α por classe — modelo TEXTURE:")
    for cls in sorted(set(classes_texture)):
        vals = [alphas_texture[i] for i, c in enumerate(classes_texture) if c == cls]
        print(f"    {cls:<12}: μ={np.mean(vals):.3f}  σ={np.std(vals):.3f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Gera figura do comportamento do roteador MoE')
    parser.add_argument('--backbone', default='ibot',
                        choices=sorted(BACKBONES_MOE))
    parser.add_argument('--epocas', type=int, default=30,
                        help='épocas de treino por tarefa (padrão: 30)')
    parser.add_argument('--fold', type=int, default=1,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument('--lambda_l2', type=float, default=0.3,
                        help='penalidade L2 nos logits do router (padrão: 0.3)')
    parser.add_argument('--por_classe', action='store_true',
                        help='gera também a figura de distribuição α por classe')
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo : {dispositivo}")
    print(f"Backbone    : {args.backbone}")
    print(f"Épocas      : {args.epocas}")
    print(f"Fold        : {args.fold}")

    alphas_por_atributo = {}
    classes_por_atributo = {}  # para análise por classe

    for atributo in ['color', 'texture']:
        print(f"\n{'='*55}")
        print(f"  Treinando MoE → {atributo.upper()} (fold {args.fold})")
        print(f"{'='*55}")

        cam_tr, rot_tr = listar_fold(atributo, args.fold, 'train')
        cam_te, rot_te = listar_fold(atributo, args.fold, 'test')
        print(f"  treino: {len(cam_tr)} imgs  |  teste: {len(cam_te)} imgs")

        ds_tr = ConjuntoDados(cam_tr, rot_tr, pegar_transformacoes(treino=True))
        ds_te = ConjuntoDados(cam_te, rot_te, pegar_transformacoes(treino=False))

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
        # mapeamento idx → nome de classe (invertido do dataset)
        idx_para_classe = {v: k for k, v in ds_te.c2i.items()}
        alphas, classes_str = coletar_alphas_com_classes(
            modelo, loader_te, dispositivo, idx_para_classe)
        alphas_por_atributo[atributo] = alphas
        classes_por_atributo[atributo] = classes_str
        print(f"  α médio: {alphas.mean():.3f} ± {alphas.std():.3f}")

    print(f"\n{'='*55}")
    print("  Gerando figuras...")
    print(f"{'='*55}")

    # Figura 5 do artigo: violino comparando o modelo treinado em cor e o em textura
    gerar_figura(
        alphas_color=alphas_por_atributo['color'],
        alphas_texture=alphas_por_atributo['texture'],
        backbone=args.backbone,
        fold=args.fold,
        pasta_saida=PASTA_FIGURAS,
    )

    # complementar: mesma distribuicao quebrada por classe
    if args.por_classe:
        gerar_figura_por_classe(
            alphas_color=alphas_por_atributo['color'],
            classes_color=classes_por_atributo['color'],
            alphas_texture=alphas_por_atributo['texture'],
            classes_texture=classes_por_atributo['texture'],
            backbone=args.backbone,
            fold=args.fold,
            pasta_saida=PASTA_FIGURAS,
        )

    print("\nFiguras geradas com sucesso!")


if __name__ == '__main__':
    main()
