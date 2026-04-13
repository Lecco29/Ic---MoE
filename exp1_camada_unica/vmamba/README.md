# VMamba para Classificação de Atributos de Roupas

Iniciação Científica

Orientadores: Alceu Britto e Arlete Beuren

## O que é esse projeto?

Esse projeto usa o Visual Mamba para classificar atributos de roupas, como cor e textura sem ficar retreinando o modelo. A ideia é extrair features de diferentes camadas do modelo e ver qual camada é melhor para cada tipo de atributo.

## Como funciona?

O VMamba tem 4 estágios. Cada estágio "enxerga" a imagem de um jeito diferente:

- stage 1: Vê detalhes básicos, tais como bordas e cores
- Stage 2: Vê padrões simples como texturas e formas
- Stage 3: Vê padrões mais complexos
- Stage 4: Vê o semântica do objeto como um todo

A gente extrai as features de cada estágio e usa um k-NN para classificar. Assim dá pra comparar qual estágio funciona melhor pra cada tarefa e nao precisa ficar retreinando o modelo.

## Dataset

O dataset tem 2000 imagens de roupas divididas em dois grupos. em cor, são 1000 imagens com 10 classes: amarillo, azul, blanco, gris, marron, naranja, negro, rojo, rosa e verde e para textura, são 1000 imagens com 10 classes: agryle, camo, checker, floral, houndstooth, leopard, paisley, plaid, solid e stripes.

As imagens foram redimensionadas para 224x224 e normalizadas com os valores do ImageNet.

## Protocolo Experimental

### Modelo

- VMamba-Tiny pré-treinado no ImageNet-1K (82.6% top-1)
- Pesos do HuggingFace: `saurabhati/VMamba_ImageNet_82.6`

### Extração de Features

- Features extraídas dos 4 estágios usando hooks
- Global Average Pooling para reduzir dimensão espacial
- Dimensões: Stage1=192, Stage2=384, Stage3=768, Stage4=768

### Classificação

- Algoritmo: k-NN com parametros assim: k=5, distância euclidiana
- Validação: 5-fold Stratified Cross-Validation
- Em cada fold: 70% treino, 30% teste

### Métricas

- Acurácia média em %
- Desvio padrão entre os folds
- F1-Score 


## Como rodar

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo/project
```

### 2. Criar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Baixar o modelo se não tiver
```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('saurabhati/VMamba_ImageNet_82.6', 'model.safetensors', local_dir='.')"
```

### 5. Rodar o projeto
```bash
python3 experimentoFinal.py
```


## Requisitos

- Python 3
- GPU com CUDA 
- ~6GB de RAM

## Referências

- [VMamba: Visual State Space Model](https://github.com/MzeroMiko/VMamba)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)



