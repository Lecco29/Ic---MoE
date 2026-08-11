#!/usr/bin/env python3
"""
Testes estatísticos de Wilcoxon signed-rank para comparar métodos/camadas.

Dependência de dados:
  - Exp01 por-fold: experiments/exp01_camada_unica/results/*_por_fold.csv
    → gerado ao re-rodar: python experiments/exp01_camada_unica/run.py
  - Exp02 por-fold: experiments/exp02_concatenacao/results/fusao_por_fold.csv
    → gerado ao re-rodar: python experiments/exp02_concatenacao/run.py
  - Exp03 por-fold: experiments/exp03_moe/results/*.csv  (já disponível)

Comparações realizadas:
  A) Entre arquiteturas (Exp01) — melhor camada de cada backbone
  B) Fusão Z vs. melhor camada individual (Exp01 vs Exp02)
  C) MoE vs. melhor camada individual (Exp01 vs Exp03)

Uso:
  python outputs/analise_wilcoxon.py
  python outputs/analise_wilcoxon.py --alfa 0.05
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

PASTA_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ    = os.path.dirname(PASTA_SCRIPT)
sys.path.insert(0, PASTA_RAIZ)

from src.backbones import BACKBONES, MELHORES_CAMADAS

PASTA_EXP1    = os.path.join(PASTA_RAIZ, 'experiments', 'exp01_camada_unica', 'results')
PASTA_EXP2    = os.path.join(PASTA_RAIZ, 'experiments', 'exp02_concatenacao', 'results')
PASTA_EXP3    = os.path.join(PASTA_RAIZ, 'experiments', 'exp03_moe', 'results')
PASTA_SAIDA   = os.path.join(PASTA_SCRIPT, 'wilcoxon')

ATRIBUTOS = ['color', 'texture']


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def wilcoxon_safe(a, b, alternativa='two-sided'):
    """
    Executa o teste de Wilcoxon signed-rank entre dois arrays de 5 valores.
    Retorna (estatística, p-valor). Retorna (nan, nan) se não houver diferenças.
    """
    d = np.array(a) - np.array(b)
    if np.all(d == 0):
        return np.nan, 1.0
    try:
        stat, p = stats.wilcoxon(a, b, alternative=alternativa)
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def sig_label(p, alfa):
    if np.isnan(p):
        return 'n/a'
    if p < 0.01:
        return '**'
    if p < alfa:
        return '*'
    return 'ns'


def carregar_exp01_por_fold(backbone, atributo):
    """Retorna dict {camada: array(5,)} com acurácias por fold."""
    arq = os.path.join(PASTA_EXP1, f'{backbone}_{atributo}_por_fold.csv')
    if not os.path.exists(arq):
        return None
    df = pd.read_csv(arq)
    resultado = {}
    for _, row in df.iterrows():
        camada = row['camada']
        accs = np.array([row[f'fold{i}'] for i in range(1, 6)])
        resultado[camada] = accs
    return resultado


def carregar_exp02_por_fold(backbone, atributo):
    """Retorna dict {'E': array(5,), 'D': array(5,), 'Z': array(5,)} ou None."""
    arq = os.path.join(PASTA_EXP2, 'fusao_por_fold.csv')
    if not os.path.exists(arq):
        return None
    df = pd.read_csv(arq)
    sub = df[(df['backbone'] == backbone) & (df['atributo'] == atributo)].sort_values('fold')
    if len(sub) != 5:
        return None
    return {
        'E': sub['acc_E'].values,
        'D': sub['acc_D'].values,
        'Z': sub['acc_Z'].values,
    }


def carregar_exp03_por_fold(backbone, atributo):
    """Retorna array(5,) com acurácias do MoE por fold."""
    arq = os.path.join(PASTA_EXP3, f'{backbone}_{atributo}.csv')
    if not os.path.exists(arq):
        return None
    df = pd.read_csv(arq).sort_values('fold')
    if len(df) < 5:
        return None
    return df['acuracia'].values[:5]


# --------------------------------------------------------------------------
# Análise A — entre arquiteturas no Exp01
# --------------------------------------------------------------------------

def analise_entre_arquiteturas(alfa):
    print("\n" + "="*70)
    print("A) ENTRE ARQUITETURAS — melhor camada de cada backbone (Exp01)")
    print("="*70)

    linhas = []
    dados_faltando = []

    for atributo in ATRIBUTOS:
        print(f"\n  Atributo: {atributo.upper()}")

        # carrega melhor camada de cada backbone
        accs_backbone = {}
        for bb in BACKBONES:
            d = carregar_exp01_por_fold(bb, atributo)
            if d is None:
                dados_faltando.append(f'{bb}_{atributo}_por_fold.csv')
                continue
            melhor = MELHORES_CAMADAS[bb][atributo]
            if melhor not in d:
                print(f"  AVISO: camada '{melhor}' não encontrada para {bb}")
                continue
            accs_backbone[bb] = d[melhor]
            print(f"    {bb:<10} ({melhor}): {d[melhor].mean():.2f}% ± {d[melhor].std():.2f}%")

        if len(accs_backbone) < 2:
            print("  → dados insuficientes para comparação")
            continue

        print(f"\n  Pares (Wilcoxon two-sided, α={alfa}):")
        for bb1, bb2 in combinations(accs_backbone.keys(), 2):
            a1, a2 = accs_backbone[bb1], accs_backbone[bb2]
            stat, p = wilcoxon_safe(a1, a2)
            label = sig_label(p, alfa)
            diff = a1.mean() - a2.mean()
            print(f"    {bb1:<10} vs {bb2:<10}: Δ={diff:+.2f}%  p={p:.4f}  {label}")
            linhas.append({
                'analise': 'A_entre_arquiteturas',
                'atributo': atributo,
                'metodo1': bb1, 'media1': a1.mean(),
                'metodo2': bb2, 'media2': a2.mean(),
                'diferenca': diff,
                'wilcoxon_stat': stat, 'p_valor': p, 'significativo': label,
            })

    if dados_faltando:
        print(f"\n  DADOS FALTANDO (re-rodar exp01): {', '.join(set(dados_faltando))}")

    return linhas


# --------------------------------------------------------------------------
# Análise B — Fusão Z vs. melhor camada individual (Exp01 vs Exp02)
# --------------------------------------------------------------------------

def analise_fusao_vs_individual(alfa):
    print("\n" + "="*70)
    print("B) FUSÃO Z vs. MELHOR CAMADA INDIVIDUAL (Exp01 vs Exp02)")
    print("="*70)

    linhas = []

    for atributo in ATRIBUTOS:
        print(f"\n  Atributo: {atributo.upper()}")
        for bb in BACKBONES:
            d01 = carregar_exp01_por_fold(bb, atributo)
            d02 = carregar_exp02_por_fold(bb, atributo)

            if d01 is None or d02 is None:
                print(f"    {bb}: dados faltando (re-rodar exp01 e/ou exp02)")
                continue

            melhor = MELHORES_CAMADAS[bb][atributo]
            if melhor not in d01:
                continue

            # melhor individual é E para cor, D para textura
            acc_individual = d01[melhor]
            acc_Z = d02['Z']

            stat, p = wilcoxon_safe(acc_individual, acc_Z)
            label = sig_label(p, alfa)
            diff = acc_Z.mean() - acc_individual.mean()

            print(f"    {bb:<10}: individual={acc_individual.mean():.2f}%  "
                  f"Z={acc_Z.mean():.2f}%  Δ={diff:+.2f}%  p={p:.4f}  {label}")
            linhas.append({
                'analise': 'B_fusao_vs_individual',
                'backbone': bb, 'atributo': atributo,
                'media_individual': acc_individual.mean(),
                'media_Z': acc_Z.mean(),
                'diferenca': diff,
                'wilcoxon_stat': stat, 'p_valor': p, 'significativo': label,
            })

    return linhas


# --------------------------------------------------------------------------
# Análise C — MoE vs. melhor camada individual (Exp01 vs Exp03)
# --------------------------------------------------------------------------

def analise_moe_vs_individual(alfa):
    print("\n" + "="*70)
    print("C) MoE vs. MELHOR CAMADA INDIVIDUAL (Exp01 vs Exp03)")
    print("="*70)

    linhas = []

    for atributo in ATRIBUTOS:
        print(f"\n  Atributo: {atributo.upper()}")
        for bb in BACKBONES:
            d01  = carregar_exp01_por_fold(bb, atributo)
            acc_moe = carregar_exp03_por_fold(bb, atributo)

            if d01 is None:
                print(f"    {bb}: dados faltando (re-rodar exp01)")
                continue
            if acc_moe is None:
                print(f"    {bb}: resultados MoE não encontrados em exp03/results/")
                continue

            melhor = MELHORES_CAMADAS[bb][atributo]
            if melhor not in d01:
                continue

            acc_individual = d01[melhor]
            stat, p = wilcoxon_safe(acc_moe, acc_individual)
            label = sig_label(p, alfa)
            diff = acc_moe.mean() - acc_individual.mean()

            print(f"    {bb:<10}: individual={acc_individual.mean():.2f}%  "
                  f"MoE={acc_moe.mean():.2f}%  Δ={diff:+.2f}%  p={p:.4f}  {label}")
            linhas.append({
                'analise': 'C_moe_vs_individual',
                'backbone': bb, 'atributo': atributo,
                'media_individual': acc_individual.mean(),
                'media_moe': acc_moe.mean(),
                'diferenca': diff,
                'wilcoxon_stat': stat, 'p_valor': p, 'significativo': label,
            })

    return linhas


# --------------------------------------------------------------------------
# Tabela formatada para o paper (LaTeX)
# --------------------------------------------------------------------------

def imprimir_tabela_latex(linhas_C):
    """Gera tabela LaTeX com resultados MoE vs individual para incluir no paper."""
    if not linhas_C:
        return

    print("\n" + "="*70)
    print("TABELA LATEX — MoE vs. Individual (para paper)")
    print("="*70)

    for atributo in ATRIBUTOS:
        print(f"\n% {atributo.upper()}")
        print(r"\begin{tabular}{lrrrrr}")
        print(r"\hline")
        print(r"Backbone & Individual (\%) & MoE (\%) & $\Delta$ & $p$-value & Sig. \\")
        print(r"\hline")
        for r in linhas_C:
            if r['atributo'] != atributo:
                continue
            p_str = f"{r['p_valor']:.4f}" if not np.isnan(r['p_valor']) else "n/a"
            print(f"{r['backbone']:<10} & "
                  f"{r['media_individual']:.2f} & "
                  f"{r['media_moe']:.2f} & "
                  f"{r['diferenca']:+.2f} & "
                  f"{p_str} & "
                  f"{r['significativo']} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Testes de Wilcoxon signed-rank')
    parser.add_argument('--alfa', type=float, default=0.05, help='nível de significância (padrão: 0.05)')
    args = parser.parse_args()

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    print(f"Nível de significância α = {args.alfa}")
    print(f"  ** = p < 0.01   * = p < {args.alfa}   ns = não significativo")
    print(f"  Nota: com n=5 folds, p mínimo (two-sided) ≈ 0.063")

    todos = []

    linhas_A = analise_entre_arquiteturas(args.alfa)
    linhas_B = analise_fusao_vs_individual(args.alfa)
    linhas_C = analise_moe_vs_individual(args.alfa)

    todos.extend(linhas_A)
    todos.extend(linhas_B)
    todos.extend(linhas_C)

    imprimir_tabela_latex(linhas_C)

    if todos:
        arq_saida = os.path.join(PASTA_SAIDA, 'resultados_wilcoxon.csv')
        pd.DataFrame(todos).to_csv(arq_saida, index=False)
        print(f"\nResultados salvos em: {arq_saida}")
    else:
        print("\nNenhum resultado gerado. Re-rodar exp01 e exp02 para gerar dados por-fold.")
        print("  python experiments/exp01_camada_unica/run.py")
        print("  python experiments/exp02_concatenacao/run.py")


if __name__ == '__main__':
    main()
