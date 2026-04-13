# script pra criar os folds do protocolo 70/30
# divide as imagens em 70% treino e 30% teste
# cria 5 folds com seeds diferentes pra ter reproducibilidade

import os
from collections import defaultdict
from sklearn.model_selection import train_test_split


def criarFolds(pastaImagens, pastaSaida, tipoAtributo, numFolds=5, seed=42):
    """
    cria os arquivos de fold pra validacao cruzada
    cada fold tem 70% das imagens pra treino e 30% pra teste
    usa stratified split pra manter a proporcao das classes
    """
    
    # cria a pasta de saida se nao existir
    os.makedirs(pastaSaida, exist_ok=True)
    
    # dicionario pra agrupar imagens por classe
    imagensPorClasse = defaultdict(list)
    
    # lista todas as imagens da pasta
    todasImagens = [f for f in os.listdir(pastaImagens) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # filtra as imagens pelo prefixo do tipo de atributo
    for nomeImagem in todasImagens:
        # pula se nao for do tipo certo
        if not nomeImagem.startswith(f"{tipoAtributo}_"):
            continue
        
        # extrai a classe do nome da imagem
        # formato do nome: tipoAtributo_classe_nomeOriginal
        partes = nomeImagem.split('_')
        if len(partes) < 3:
            continue
        
        classe = partes[1]  # a classe eh a segunda parte
        
        # monta o caminho original (formato do dataset original)
        nomeOriginal = '_'.join(partes[2:])
        caminhoOriginal = f"/mnt/data/fabric/dataset_atributos/{tipoAtributo}/{classe}/{nomeOriginal}"
        
        # guarda a imagem na lista da classe
        imagensPorClasse[classe].append((classe, caminhoOriginal, nomeImagem))
    
    # ordena as classes pra ficar organizado
    classes = sorted(imagensPorClasse.keys())
    
    print(f"encontradas {len(classes)} classes: {classes}")
    
    # mostra quantas imagens tem em cada classe
    for classe in classes:
        print(f"  {classe}: {len(imagensPorClasse[classe])} imagens")
    
    # avisa se alguma classe nao tiver 100 imagens (esperado pelo dataset)
    for classe, imagens in imagensPorClasse.items():
        if len(imagens) != 100:
            print(f"AVISO: classe {classe} tem {len(imagens)} imagens (esperado 100)")
    
    # cria cada um dos 5 folds
    for numFold in range(1, numFolds + 1):
        print(f"\ncriando fold {numFold}...")
        
        # usa seed diferente pra cada fold (seed base + numero do fold)
        # isso garante que os folds sejam reproduziveis
        seedFold = seed + numFold
        
        listaTreino = []
        listaTeste = []
        
        # divide cada classe separadamente (stratified)
        for classe in classes:
            imagensDaClasse = imagensPorClasse[classe].copy()
            
            # divide 70% treino 30% teste usando sklearn
            treino, teste = train_test_split(
                imagensDaClasse,
                test_size=0.3,
                random_state=seedFold,
                shuffle=True
            )
            
            # formata as linhas no padrao: indice;classe;caminho
            for idx, (cl, caminho, img) in enumerate(treino):
                listaTreino.append(f"{idx};{cl};{caminho}\n")
            
            for idx, (cl, caminho, img) in enumerate(teste):
                listaTeste.append(f"{idx};{cl};{caminho}\n")
            
            print(f"  {classe}: {len(treino)} treino, {len(teste)} teste")
        
        # salva os arquivos do fold
        arquivoTreino = os.path.join(pastaSaida, f'fold{numFold}-train.txt')
        arquivoTeste = os.path.join(pastaSaida, f'fold{numFold}-test.txt')
        
        with open(arquivoTreino, 'w') as f:
            f.writelines(listaTreino)
        
        with open(arquivoTeste, 'w') as f:
            f.writelines(listaTeste)
        
        print(f"  arquivos salvos: {len(listaTreino)} treino, {len(listaTeste)} teste")
    
    print(f"\ntodos os folds criados em: {pastaSaida}")


def main():
    """
    funcao principal - cria os folds pra cor e textura
    """
    
    PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
    pastaDados = os.path.join(PASTA_RAIZ, 'data')
    
    print("=" * 50)
    print("CRIACAO DOS FOLDS - PROTOCOLO 70/30")
    print("=" * 50)
    
    # cria folds pro dataset de cor
    print("\n--- DATASET DE COR ---")
    criarFolds(
        pastaImagens=os.path.join(pastaDados, 'images', 'color'),
        pastaSaida=os.path.join(pastaDados, 'Protocolo', 'folds_color_70_30', 'folds'),
        tipoAtributo='color',
        numFolds=5,
        seed=42
    )
    
    # cria folds pro dataset de textura
    print("\n--- DATASET DE TEXTURA ---")
    criarFolds(
        pastaImagens=os.path.join(pastaDados, 'images', 'texture'),
        pastaSaida=os.path.join(pastaDados, 'Protocolo', 'folds_texture_70_30', 'folds'),
        tipoAtributo='texture',
        numFolds=5,
        seed=42
    )
    
    print("\nconcluido!")


if __name__ == '__main__':
    main()
