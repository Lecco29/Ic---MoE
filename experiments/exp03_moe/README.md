# Experimento 3 — MoE

A ideia aqui veio dos resultados do exp01: camadas rasas pegam melhor cor, camadas profundas pegam melhor textura. No exp02 a gente concatenou as duas e funcionou, mas o peso é fixo pra todas as imagens. O MoE tenta aprender esse balanço por imagem — dependendo da lesão, talvez cor seja mais útil, em outra talvez textura.

## Arquitetura

O backbone extrai duas representações:

```
fE_raw  (camada rasa,  ex: block2 no iBOT)
fD_raw  (camada deep,  ex: block9 no iBOT)
```

Duas projeções lineares (`hE`, `hD`) jogam tudo pra 256d. O router recebe os dois vetores concatenados e devolve dois pesos (α, β) que somam 1:

```
z = α · fE + β · fD
```

Se a imagem tem cor forte: α sobe. Se textura é o que discrimina: β sobe. Isso é aprendido durante o treino, não definido à mão.

## Router

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
        g = torch.cat([fE, fD], dim=1)
        weights = torch.softmax(self.mlp(g), dim=1)
        return weights[:, 0:1], weights[:, 1:2]  # alpha, beta
```

## Componentes (models/moe.py)

- `CabecaProjecao` — Linear + LayerNorm + ReLU, dim=256
- `Router` — MLP [2d → 128 → 2] + Softmax
- `BackboneIBOT` — ViT-S/16 (timm), early=block2, deep=block9
- `BackboneResNet50` — early=layer1 (256d), deep=layer3 (1024d)
- `BackboneVGG16` — early=pool1 (64d), deep=pool3 (256d)
- `BackboneVMamba` — early=stage1 (192d), deep=stage2 (384d)
- `ModeloMoE` — junta tudo

## Como usar

```bash
python run.py --backbone ibot --atributo color
python run.py --backbone resnet50 --atributo texture
python run.py --backbone vmamba --atributo both --epocas 50
```

Args:
- `--backbone`: ibot | resnet50 | vgg16 | vmamba
- `--atributo`: color | texture | both
- `--epocas`: padrão 30
- `--d`: dim das projeções, padrão 256
- `--lote`: batch size, padrão 32
- `--lr`: padrão 1e-4
- `--k`: k do KNN, padrão 5
- `--so_avaliar`: pula o treino

## Protocolo

5-fold cross-validation, 70/30. Backbone congelado, só hE + hD + router treinam. 30 épocas, AdamW + CosineAnnealingLR. KNN k=5 com StandardScaler nos embeddings z.
