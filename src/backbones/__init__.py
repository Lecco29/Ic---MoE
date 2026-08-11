"""Extratores de features dos backbones avaliados no artigo.

Todos expoem a mesma interface:
    extrator.extrairFeatures(batch)  -> {nome_da_camada: tensor [B, D]}
    extrator.pegarDimensoes()        -> {nome_da_camada: D}

O iBOT tem ainda extrairFeaturesComCLS(), usada nos experimentos.
"""

BACKBONES = ('vgg16', 'resnet50', 'ibot', 'vmamba')

# Melhor camada de cada backbone por atributo, medida no exp01 (Tabela III do
# artigo). E o ponto de partida do exp02 e das figuras de retrieval:
# a de cor e sempre rasa, a de textura sempre mais profunda.
MELHORES_CAMADAS = {
    'vgg16':    {'color': 'layer1', 'texture': 'layer4'},
    'resnet50': {'color': 'layer2', 'texture': 'layer3'},
    'ibot':     {'color': 'block2', 'texture': 'block9'},
    'vmamba':   {'color': 'stage1', 'texture': 'stage2'},
}


def criar_extrator(nome, dispositivo='auto'):
    """Instancia o extrator de um backbone pelo nome.

    Os imports sao locais porque cada backbone puxa dependencias pesadas
    (e o VMamba so funciona com os pesos baixados).
    """
    if nome == 'vgg16':
        from .vgg16 import criarExtrator
        return criarExtrator(dispositivo=dispositivo)

    if nome == 'resnet50':
        from .resnet50 import criarExtrator
        return criarExtrator(dispositivo=dispositivo)

    if nome == 'ibot':
        from .ibot import criarExtrator
        return criarExtrator(modelo='vit_small', dispositivo=dispositivo)

    if nome == 'vmamba':
        from .vmamba import criarExtrator
        return criarExtrator(dispositivo=dispositivo)

    raise ValueError(f"backbone desconhecido: {nome} (use um de {BACKBONES})")
