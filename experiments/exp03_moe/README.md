# Experimento 3 — MoE (Mixture of Experts)

Fusão adaptativa de features usando um router que aprende, por imagem, quanto peso dar
para a representação de cor (camada rasa) vs textura (camada profunda).

## Motivação

Os experimentos 1 e 2 mostraram que camadas rasas capturam melhor cor e camadas profundas
capturam melhor textura. Em vez de concatenar fixo (Exp02) ou escolher uma só (Exp01), o MoE
aprende esse balanço automaticamente — por imagem, não globalmente.

## Arquitetura (sugestão do Prof. Alceu Britto Jr.)

Dado uma imagem `x`, o backbone `B(x)` extrai três embeddings de camadas diferentes:

```
B(x) → fE_raw   (camada rasa  — early, ex: Block 2 no iBOT)
      → fD_raw   (camada deep  — ex: Block 9 no iBOT)
      → fL_raw   (última camada)

hE(fE_raw) → fE  [256d]
hD(fD_raw) → fD  [256d]

Router(fE, fD) → α, β      (α + β = 1)

z = α · fE + β · fD
```

`hE` e `hD` são projeções lineares que padronizam os embeddings para o mesmo tamanho (256d),
independente do backbone usado. O router recebe `fE` e `fD` concatenados e decide os pesos.

## Router

Código base fornecido pelo Prof. Alceu Britto Jr.:

```python
class Router(nn.Module):
    def __init__(self, d=256, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*d, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

    def forward(self, fE, fD):
        g = torch.cat([fE, fD], dim=1)   # [B, 2d]
        logits = self.mlp(g)              # [B, 2]
        weights = torch.softmax(logits, dim=1)
        alpha = weights[:, 0:1]           # peso para cor
        beta  = weights[:, 1:2]           # peso para textura
        return alpha, beta
```

Se a imagem tem cor como atributo dominante: α alto, β baixo. Se textura for mais
discriminativa: α baixo, β alto. O router aprende isso durante o treino.

## Componentes (models/moe.py)

- `CabecaProjecao` — Linear + LayerNorm + ReLU, projeta para d=256
- `Router` — MLP [2d → 128 → 2] + Softmax
- `BackboneIBOT` — ViT-S/16, early=block2, deep=block9
- `BackboneResNet50` — early=layer1 (256d), deep=layer3 (1024d)
- `BackboneVGG16` — early=pool1 (64d), deep=pool3 (256d)
- `BackboneVMamba` — early=stage1 (192d), deep=stage2 (384d)
- `ModeloMoE` — backbone + hE + hD + router

## Como usar

```bash
python run.py --backbone ibot --atributo color
python run.py --backbone resnet50 --atributo texture
python run.py --backbone vmamba --atributo both --epocas 50
```

Argumentos:
- `--backbone`: ibot | resnet50 | vgg16 | vmamba
- `--atributo`: color | texture | both
- `--epocas`: número de épocas (padrão: 30)
- `--d`: dimensão das projeções hE e hD (padrão: 256)
- `--lote`: batch size (padrão: 32)
- `--lr`: learning rate (padrão: 1e-4)
- `--k`: k do KNN (padrão: 5)
- `--so_avaliar`: pula o treino, útil pra debug

## Protocolo

- 5-fold cross-validation, 70% treino / 30% teste
- Backbone congelado — só hE, hD e router treinam
- 30 épocas, AdamW lr=1e-4, CosineAnnealingLR
- Avaliação final: KNN k=5 com StandardScaler nos embeddings z

## Resultados (acurácia média, 5 folds)

| Backbone  | Cor (%)       | Textura (%)   |
|-----------|---------------|---------------|
| iBOT      | 97.33 ± 0.56  | 95.47 ± 0.96  |
| ResNet50  | 97.27 ± 0.71  | 97.67 ± 0.47  |
| VGG16     | 93.40 ± 0.71  | 96.13 ± 1.00  |
| VMamba    | 96.93 ± 1.02  | 98.00 ± 0.73  |
