# Layer-Wise Attribute Sensitivity and Adaptive Fusion with Mixture of Experts

Código e resultados do artigo submetido ao **IECON 2026**:

> **Layer-Wise Attribute Sensitivity and Adaptive Fusion with Mixture of Experts in Deep Models for Color and Texture Classification**
> Arlete T. Beuren, Leonardo J. R. Pinto, Jonathan de Matos, Jean P. Barddal, Alceu S. Britto Jr.
> UTFPR Santa Helena · UEPG · PUCPR

O PDF submetido está em [`paper/iecon2026_submetido.pdf`](paper/iecon2026_submetido.pdf).

## O que o trabalho investiga

Classificação e recuperação de imagens de roupas por **cor** e por **textura**,
1.000 imagens e 10 classes para cada atributo. A hipótese central é que camadas
rasas capturam cor e camadas profundas capturam textura — e que isso vale tanto
para CNNs quanto para Vision Transformers e State Space Models.

Três experimentos:

1. **Camada única** — cada camada isolada, avaliada com kNN. Confirma a inversão
   entre as curvas de cor e textura.
2. **Concatenação** — junta a melhor camada de cor com a melhor de textura.
   Geralmente não ajuda; em cor chega a piorar bastante.
3. **MoE** — um router aprende, imagem a imagem, quanto usar de cada camada.
   É a contribuição principal, e supera as duas alternativas na maioria dos casos.

## Estrutura

```
src/
  backbones/          extratores dos 4 backbones, mesma interface
    vmamba_hf/        VMamba do HuggingFace, vendorizado (MIT)
  dados.py            folds, pré-processamento e extração em lote
  evaluation/         mAP@10, R@1, R@5
experiments/
  exp01_camada_unica/ run.py + results/
  exp02_concatenacao/ run.py + results/
  exp03_moe/          run.py, models/moe.py, gerar_figura_router.py, results/
outputs/
  gerar_curvas_desempenho.py   curvas de acurácia por profundidade
  visualizar_retrieval.py      grades query -> top-5
  analise_wilcoxon.py          testes de significância
  figures/, wilcoxon/          saídas dos scripts acima
paper/                fonte LaTeX, figuras e o PDF submetido
data/                 protocolo dos folds (imagens à parte, ver data/README.md)
checkpoints/          pesos dos modelos (à parte, ver checkpoints/README.md)
```

## De onde vem cada número do artigo

| No artigo | Script | Resultado |
|---|---|---|
| Tabela III (melhor camada) | `experiments/exp01_camada_unica/run.py` | `results/{backbone}_{atributo}.csv` |
| Tabela V (concatenação) | `experiments/exp02_concatenacao/run.py` | `results/fusao.csv` |
| Tabelas VI e VII (MoE) | `experiments/exp03_moe/run.py` | `results/{backbone}_{atributo}.csv` |
| Tabela VIII (mAP@10, R@1, R@5) | os três acima | colunas `map10_*`, `r1_*`, `r5_*` |
| Figura 2 (acurácia × profundidade) | `outputs/gerar_curvas_desempenho.py` | `outputs/figures/curvas_desempenho_profundidade.png` |
| Figuras 3 e 4 (retrieval) | `outputs/visualizar_retrieval.py` | `outputs/figures/retrieval/ibot_*_top5.pdf` |
| Figura 5 (pesos do router) | `experiments/exp03_moe/gerar_figura_router.py` | `outputs/figures/fig_router_behavior.pdf` |

**Tabela IV (LMFCN) não é reproduzível aqui.** Esses resultados vieram do método
de de Matos et al. (referência [19] do artigo) e o código não faz parte deste
repositório.

## Como rodar

Requer as imagens em `data/images/` e os pesos em `checkpoints/`.

```bash
pip install -r requirements.txt

python experiments/exp01_camada_unica/run.py                    # todos os backbones
python experiments/exp02_concatenacao/run.py
python experiments/exp03_moe/run.py --backbone ibot --atributo both

python outputs/gerar_curvas_desempenho.py
python outputs/visualizar_retrieval.py --backbone ibot
python outputs/analise_wilcoxon.py
python experiments/exp03_moe/gerar_figura_router.py --backbone ibot
```

Cada script aceita `--backbone` para rodar um de cada vez. Os experimentos 1 e 2
só fazem inferência; o 3 treina as projeções e o router, com o backbone congelado.

## Detalhes que valem saber

- **Protocolo**: 5 folds 70/30, kNN com k=5 e distância euclidiana. Na avaliação
  de retrieval, a galeria é o conjunto de treino, o que evita casar a query com
  ela mesma.
- **iBOT no experimento 3**: o exp03 usa o ViT-S/16 **supervisionado** do `timm`,
  enquanto os exp01 e exp02 usam o checkpoint **auto-supervisionado** do iBOT.
  Foi assim que os números publicados do MoE saíram; trocar o checkpoint muda os
  resultados. Está anotado em `experiments/exp03_moe/models/moe.py`.
- **Fonte LaTeX**: `paper/iecon2026_en.tex` é a versão local mais próxima do que
  foi submetido, não uma cópia exata. A revisão final foi feita fora deste
  repositório e traz parágrafos reescritos e uma lista de autores diferente. Em
  caso de divergência, vale o PDF.
