# Dados

## Protocolo (versionado)

`protocols/` traz os 5 folds usados no artigo, 70% treino / 30% teste, separados
por atributo. Cada linha é `indice;rotulo;caminho_original`, onde o caminho
aponta para a máquina em que o dataset foi montado — o que importa dele é só o
nome do arquivo.

## Imagens (não versionadas)

`images/color/` e `images/texture/`, 1.000 imagens cada, 10 classes por atributo.
Somam 270 MB e por isso ficam fora do repositório.

Os arquivos são gravados achatados numa pasta só, no padrão
`{atributo}_{rotulo}_{arquivo}` — por exemplo `color_amarillo_15626.jpg`. É esse
nome que `src/dados.py` monta a partir de cada linha do fold.

O dataset é o de Baloian et al., referência [2] do artigo.
