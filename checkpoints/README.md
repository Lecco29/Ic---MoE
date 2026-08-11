# Pesos dos modelos

Os arquivos de peso não vão no repositório (somam ~200 MB). Coloque-os aqui
antes de rodar os experimentos.

| Arquivo | Modelo | Origem |
|---|---|---|
| `checkpoint_ibot_vits16.pth` | iBOT ViT-S/16 auto-supervisionado | [bytedance/ibot](https://github.com/bytedance/ibot) — checkpoint ViT-S/16 pré-treinado no ImageNet-1K |
| `model.safetensors` | VMamba-Tiny | port HuggingFace do VMamba, mesmo repositório de onde veio `src/backbones/vmamba_hf/` |

VGG-16 e ResNet-50 não precisam de download manual: os pesos do ImageNet vêm
pelo `torchvision` na primeira execução.

O backbone iBOT do experimento 3 é um caso à parte — ele usa o ViT-S/16
supervisionado do `timm`, baixado automaticamente. Veja a nota em
`experiments/exp03_moe/models/moe.py`.
