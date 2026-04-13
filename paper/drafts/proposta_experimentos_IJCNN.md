# Proposta de Experimentos - IJCNN 2026

## Questões de Pesquisa

**RQ1.** How does color and texture information distribute across the depth of CNN, ViT, and Mamba architectures for visual classification tasks?

**RQ2.** Which network layer yields the best classification performance for color-based and texture-based tasks?

**RQ3.** Is the last-layer representation optimal for classification, or do intermediate layers provide superior performance for low-level visual attributes?

**RQ4.** Does early-deep feature fusion improve classification performance compared to single-layer representations?

**RQ5.** Are the layer-wise color and texture representation patterns consistent across CNN, Transformer, and Mamba architectures?

---

## Experimento 1: Descobrir a melhor camada para cor e textura (kNN)

**Objetivo:** Identificar como cor e textura são representadas ao longo da hierarquia das redes.

**Status:** Concluído

### Metodologia

**Etapa 1. Extração de embeddings**

Para cada backbone (CNN, ViT, Mamba):
- Extrair embeddings de camadas internas (early / mid / deep)
- Extrair embeddings da última camada
- Aplicar Global Average Pooling (GAP)

**Etapa 2. Classificação com kNN**

Para cada camada:
- Avaliar classificação de cor usando kNN (k=5)
- Avaliar classificação de textura usando kNN (k=5)
- Protocolo: 70% treino / 30% teste, 5-fold cross-validation

**Métricas:**
- Accuracy
- F1-score

**Dataset:**
- 1000 imagens cor (10 classes × 100 imagens)
- 1000 imagens textura (10 classes × 100 imagens)

**Resultados esperados:**
- Curvas desempenho × profundidade
- Para cada modelo: E = melhor camada para cor, D = melhor camada para textura

### Resultados

#### VGG-16 (CNN Clássica)

| Layer | Color Acc (%) | Color F1 | Texture Acc (%) | Texture F1 | Dim |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Layer 1 | **95.13 ± 0.81** | 95.12 | 70.73 ± 2.25 | 70.56 | 64 |
| Layer 2 | 90.27 ± 1.27 | 90.22 | 86.60 ± 1.57 | 86.61 | 128 |
| Layer 3 | 85.47 ± 2.33 | 85.35 | 93.20 ± 1.48 | 93.18 | 256 |
| Layer 4 | 71.80 ± 2.55 | 70.99 | **94.73 ± 0.74** | 94.76 | 512 |
| Layer 5 | 55.07 ± 2.64 | 55.12 | 89.40 ± 1.64 | 89.42 | 512 |

**Melhor camada cor (E):** Layer 1 (95.13%)
**Melhor camada textura (D):** Layer 4 (94.73%)

#### ResNet-50 (CNN com Skip Connections)

| Layer | Color Acc (%) | Color F1 | Texture Acc (%) | Texture F1 | Dim |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Layer 1 | 95.47 ± 0.78 | 95.48 | 83.40 ± 1.36 | 83.39 | 256 |
| Layer 2 | **95.87 ± 0.50** | 95.85 | 90.67 ± 1.65 | 90.65 | 512 |
| Layer 3 | 88.27 ± 1.12 | 88.13 | **93.27 ± 0.25** | 93.23 | 1024 |
| Layer 4 | 59.20 ± 3.11 | 59.19 | 86.33 ± 0.92 | 86.23 | 2048 |

**Melhor camada cor (E):** Layer 2 (95.87%)
**Melhor camada textura (D):** Layer 3 (93.27%)

#### iBOT (Vision Transformer)

| Block | Color Acc (%) | Color F1 | Texture Acc (%) | Texture F1 | Dim |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Block 0 | 94.53 ± 1.13 | 94.58 | 52.20 ± 2.69 | 51.60 | 384 |
| Block 1 | 95.40 ± 1.27 | 95.38 | 69.87 ± 2.28 | 69.75 | 384 |
| Block 2 | **95.87 ± 1.05** | 95.86 | 79.93 ± 2.61 | 79.69 | 384 |
| Block 3 | 95.20 ± 0.83 | 95.19 | 87.87 ± 1.44 | 87.62 | 384 |
| Block 4 | 92.13 ± 0.27 | 92.19 | 90.53 ± 1.33 | 90.40 | 384 |
| Block 5 | 88.73 ± 0.77 | 88.78 | 91.00 ± 2.02 | 90.91 | 384 |
| Block 6 | 84.93 ± 1.24 | 84.89 | 92.20 ± 1.85 | 92.07 | 384 |
| Block 7 | 84.07 ± 1.18 | 83.95 | 95.00 ± 0.99 | 94.98 | 384 |
| Block 8 | 76.00 ± 0.76 | 75.82 | 96.87 ± 0.54 | 96.86 | 384 |
| Block 9 | 69.93 ± 1.97 | 69.81 | **97.20 ± 0.54** | 97.19 | 384 |
| Block 10 | 66.13 ± 3.28 | 65.94 | 96.27 ± 0.53 | 96.25 | 384 |
| Block 11 | 62.80 ± 1.73 | 62.66 | 96.00 ± 0.87 | 95.97 | 384 |

**Melhor camada cor (E):** Block 2 (95.87%)
**Melhor camada textura (D):** Block 9 (97.20%)

#### VMamba (State Space Model)

| Stage | Color Acc (%) | Color F1 | Texture Acc (%) | Texture F1 | Dim |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage 1 | **94.41 ± 0.52** | 94.35 | 93.86 ± 1.07 | 93.33 | 192 |
| Stage 2 | 93.41 ± 0.47 | 93.33 | **95.76 ± 0.84** | 95.36 | 384 |
| Stage 3 | 78.34 ± 3.75 | 78.10 | 95.76 ± 0.75 | 95.37 | 768 |
| Stage 4 | 54.14 ± 2.39 | 53.94 | 86.52 ± 2.57 | 85.60 | 768 |

**Melhor camada cor (E):** Stage 1 (94.41%)
**Melhor camada textura (D):** Stage 2 (95.76%)

### Resumo Comparativo

| Backbone | Melhor Cor (E) | Acc Cor (%) | Melhor Textura (D) | Acc Textura (%) |
|:---:|:---:|:---:|:---:|:---:|
| VGG-16 | Layer 1 | 95.13 | Layer 4 | 94.73 |
| ResNet-50 | Layer 2 | 95.87 | Layer 3 | 93.27 |
| iBOT | Block 2 | 95.87 | Block 9 | **97.20** |
| VMamba | Stage 1 | 94.41 | Stage 2 | 95.76 |

### Curvas de Desempenho × Profundidade

![Curvas de desempenho por profundidade para todos os backbones](curvas_desempenho_profundidade.png)

*Figura 1. Curvas de desempenho × profundidade para cada backbone. Vermelho = classificação de cor, Azul = classificação de textura. Círculos/quadrados grandes indicam a melhor camada (E para cor, D para textura).*

![Comparação entre arquiteturas](curvas_comparacao_arquiteturas.png)

*Figura 2. Comparação normalizada entre arquiteturas. Esquerda: classificação de cor (early layers melhores). Direita: classificação de textura (mid/deep layers melhores).*

### Observações (Responde RQ1, RQ2, RQ3, RQ5)

**RQ1 - Distribuição de cor e textura:**
- Cor: concentrada nas camadas iniciais (early layers)
- Textura: concentrada nas camadas intermediárias/profundas (mid/deep layers)
- Padrão consistente em todas as arquiteturas

**RQ2 - Melhor camada:**
- Cor: camadas 1-2 (CNNs), blocks 0-3 (iBOT), stage 1 (VMamba)
- Textura: camadas 3-4 (CNNs), blocks 7-9 (iBOT), stages 2-3 (VMamba)

**RQ3 - Última camada é ótima?**
- NÃO. A última camada apresenta desempenho inferior para ambos os atributos
- VGG-16 Layer 5: 55.07% cor, 89.40% textura
- ResNet-50 Layer 4: 59.20% cor, 86.33% textura
- iBOT Block 11: 62.80% cor, 96.00% textura
- VMamba Stage 4: 54.14% cor, 86.52% textura

**RQ5 - Padrão consistente entre arquiteturas?**
- SIM. Todas as arquiteturas (CNN, Transformer, Mamba) apresentam o mesmo padrão:
  - Early layers → melhor para cor
  - Deep layers → melhor para textura

---

## Experimento 2: Fusão Early-Deep (Early Fusion)

**Objetivo:** Avaliar se a fusão da melhor camada de cor (E) com a melhor camada de textura (D) melhora a classificação.

**Status:** Concluído

### Metodologia

Para cada backbone:
1. Identificar E = melhor camada para cor (do Experimento 1)
2. Identificar D = melhor camada para textura (do Experimento 1)
3. Realizar fusão: `z = concat(E, D)`
4. Avaliar classificação com kNN (k=5, 5-fold CV) usando z
5. Comparar com uso de camadas isoladas

### Configuração da Fusão

| Backbone | E (Cor) | Dim E | D (Textura) | Dim D | Dim z |
|:---:|:---:|:---:|:---:|:---:|:---:|
| VGG-16 | Layer 1 | 64 | Layer 4 | 512 | 576 |
| ResNet-50 | Layer 2 | 512 | Layer 3 | 1024 | 1536 |
| iBOT | Block 2 | 384 | Block 9 | 384 | 768 |
| VMamba | Stage 1 | 96 | Stage 2 | 192 | 288 |

*Protocolo: 70% treino / 30% teste, 5-fold cross-validation, kNN (k=5)*

### Resultados - Classificação de COR

| Backbone | E (cor) | D (textura) | Z (fusão) | Melhora |
|:---:|:---:|:---:|:---:|:---:|
| VGG-16 | **95.13 ± 0.81%** | 71.80 ± 2.55% | 86.13 ± 1.86% | -9.00% |
| ResNet-50 | **95.87 ± 0.50%** | 88.27 ± 1.12% | 94.13 ± 0.78% | -1.73% |
| iBOT | **96.93 ± 0.49%** | 81.33 ± 1.62% | 85.13 ± 1.69% | -11.80% |
| VMamba | **94.80 ± 0.62%** | 93.93 ± 0.25% | 94.13 ± 0.16% | -0.67% |

### Resultados - Classificação de TEXTURA

| Backbone | E (cor) | D (textura) | Z (fusão) | Melhora |
|:---:|:---:|:---:|:---:|:---:|
| VGG-16 | 70.73 ± 2.25% | **94.73 ± 0.74%** | 94.73 ± 0.90% | 0.00% |
| ResNet-50 | 90.67 ± 1.65% | 93.27 ± 0.25% | **93.53 ± 1.29%** | +0.27% |
| iBOT | 78.33 ± 0.92% | **87.80 ± 1.36%** | 88.07 ± 1.04% | +0.27% |
| VMamba | 88.40 ± 1.51% | **94.33 ± 1.40%** | 93.40 ± 1.96% | -0.93% |

### Análise dos Resultados

**Classificação de Cor:**
- A fusão NÃO melhora o desempenho em nenhum backbone
- A camada E (otimizada para cor) sozinha é sempre melhor que a fusão
- A adição de features de textura (D) "confunde" o classificador de cor
- Maior degradação no iBOT (-11.80%)
- VMamba mostra menor degradação (-0.67%) pois suas camadas E e D têm desempenho similar para cor

**Classificação de Textura:**
- Resultados mistos: pequena melhora em ResNet-50 (+0.27%) e iBOT (+0.27%)
- VGG-16 ficou igual e VMamba teve leve degradação (-0.93%)
- A fusão não traz benefícios significativos para classificação de textura

**Observação sobre VMamba:**
- VMamba distribui informação de cor de forma mais uniforme entre camadas (94.80% vs 93.93%)
- Diferentemente de outros backbones, onde há grande diferença entre E e D para cor

### Visualizações do Experimento 2

![Comparação E vs D vs Z](exp2_fusao_comparacao.png)

*Figura 3. Comparação entre E (camada de cor), D (camada de textura) e Z (fusão) para cada backbone e tarefa.*

![Impacto da fusão](exp2_fusao_improvement.png)

*Figura 4. Impacto da fusão: diferença entre Z e a melhor camada individual (E ou D). Valores negativos indicam que a fusão piorou o desempenho.*

**Tabela Resumo - Experimento 2: Fusão (Δ = Z - best(E,D))**

| Backbone | Task | E (color) | D (texture) | Z (fusion) | Best | Δ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VGG-16 | Color | 95.1% | 71.8% | 86.1% | E | -9.0% |
| | Texture | 70.7% | 94.7% | 94.7% | D | +0.0% |
| ResNet-50 | Color | 95.9% | 88.3% | 94.1% | E | -1.7% |
| | Texture | 90.7% | 93.3% | 93.5% | D | +0.3% |
| iBOT | Color | 96.9% | 81.3% | 85.1% | E | -11.8% |
| | Texture | 78.3% | 87.8% | 88.1% | D | +0.3% |
| VMamba | Color | 94.8% | 93.9% | 94.1% | E | -0.7% |
| | Texture | 88.4% | 94.3% | 93.4% | D | -0.9% |

*Tabela: Δ negativo = fusão piorou, Δ positivo = fusão melhorou*

### Responde RQ4

**RQ4 - Fusão early-deep melhora desempenho?**

**RESPOSTA: NÃO para classificação de atributos individuais.**

A fusão simples por concatenação (early fusion) das melhores camadas de cor e textura **não melhora** o desempenho comparado com o uso da camada específica para cada atributo:

- Para **cor**: usar apenas E (camada de cor) é sempre melhor
- Para **textura**: usar apenas D (camada de textura) é geralmente melhor ou equivalente

**Implicação:** Para tarefas de classificação de atributos visuais específicos, é mais eficiente usar a camada especializada para cada atributo do que combinar representações.

---

## Conclusões Gerais

### Respostas às Questões de Pesquisa

| RQ | Pergunta | Resposta |
|:---:|:---|:---|
| RQ1 | Como cor/textura se distribuem nas camadas? | Cor nas camadas iniciais, textura nas profundas |
| RQ2 | Qual camada tem melhor desempenho? | Depende do atributo (ver tabelas Exp. 1) |
| RQ3 | Última camada é ótima? | **NÃO** - camadas intermediárias são melhores |
| RQ4 | Fusão melhora desempenho? | **NÃO** - camada específica é melhor |
| RQ5 | Padrão consistente entre arquiteturas? | **SIM** - CNN, Transformer e Mamba seguem o mesmo padrão |

### Próximos Passos

1. [x] Implementar script de fusão early-deep
2. [x] Executar Experimento 2 para todos os backbones
3. [x] Analisar resultados e responder RQ4
4. [x] Gerar visualizações (curvas desempenho × profundidade)
5. [ ] Escrever artigo final
