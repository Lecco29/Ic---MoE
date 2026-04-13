# Experimento 3 — MoE (Mixture of Experts)

Fusão adaptativa de features early (cor) e deep (textura) usando um router que aprende
quanto usar de cada camada por imagem.

## Ideia

Em vez de concatenar as features (Exp02) ou usar só uma camada (Exp01), o MoE aprende
pesos alpha e beta por imagem:

```
Backbone(x) → fE_raw (camada rasa)
             fD_raw (camada profunda)

hE(fE_raw) → fE  [256d]    projeção padronizada
hD(fD_raw) → fD  [256d]

Router(fE, fD) → α, β      pesos por imagem (α + β = 1)

z = α · fE + β · fD        vetor final
```

Se a imagem é fácil de classificar por cor: α alto, β baixo.
Se textura é mais discriminativa: α baixo, β alto.
O router aprende isso automaticamente durante o treino.

## Componentes (models/moe.py)

- `CabecaProjecao` — Linear + LayerNorm + ReLU, projeta pra dim d=256
- `Router` — MLP [2d → 128 → 2] + Softmax (código base do Prof. Alceu Britto Jr.)
- `BackboneIBOT` — ViT-S/16 via timm, early=block2, deep=block9
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

Argumentos disponíveis:
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
