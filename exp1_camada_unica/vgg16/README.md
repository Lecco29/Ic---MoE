# VGG-16 para Classificacao de Atributos de Roupas

Iniciacao Cientifica

Orientadores: Alceu Britto e Arlete Beuren

## O que e esse projeto?

Esse projeto usa a VGG-16 pre-treinada no ImageNet para classificar atributos de roupas, como cor e textura sem ficar retreinando o modelo. A ideia e extrair features de diferentes camadas do modelo e ver qual camada e melhor para cada tipo de atributo.

## Como funciona?

A VGG-16 tem 5 blocos convolucionais. Cada bloco "enxerga" a imagem de um jeito diferente:

- Camada 1: Ve detalhes basicos como bordas e cores (64 filtros)
- Camada 2: Ve padroes simples como texturas basicas (128 filtros)
- Camada 3: Ve padroes mais complexos (256 filtros)
- Camada 4: Ve padroes ainda mais complexos (512 filtros)
- Camada 5: Ve a semantica do objeto como um todo (512 filtros)

A gente extrai as features de cada camada (apos max pooling) e usa um k-NN para classificar. Assim da pra comparar qual camada funciona melhor pra cada tarefa e nao precisa ficar retreinando o modelo.

## Dataset

O dataset tem 2000 imagens de roupas divididas em dois grupos. Em cor, sao 1000 imagens com 10 classes. Para textura, sao 1000 imagens com 10 classes.

As imagens foram redimensionadas para 224x224 e normalizadas com os valores do ImageNet.

## Protocolo Experimental

### Modelo

- VGG-16 pre-treinada no ImageNet
- Pesos do torchvision (ImageNet1K_V1)

### Extracao de Features

- Features extraidas das 5 camadas de max pooling usando hooks
- Global Average Pooling para reduzir dimensao espacial
- Dimensoes: Layer1=64, Layer2=128, Layer3=256, Layer4=512, Layer5=512

### Classificacao

- Algoritmo: k-NN com k=5, distancia euclidiana
- Validacao: 5-fold com protocolo 70/30
- Em cada fold: 70% treino, 30% teste

### Metricas

- Acuracia media em %
- Desvio padrao entre os folds
- F1-Score

## Como rodar

### 1. Criar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install torch torchvision scikit-learn pandas tqdm pillow
```

### 3. Criar os folds
```bash
python3 protocolo_folds.py
```

### 4. Rodar o projeto
```bash
python3 main.py
```

## Requisitos

- Python 3
- GPU com CUDA (opcional, mas recomendado)
- ~4GB de RAM

## Referencias

- [VGG-16: Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
