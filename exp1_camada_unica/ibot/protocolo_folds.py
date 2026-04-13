import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

# essa funcao foi feita para criar os folds 70/30 para o protocolo 
def criar_folds(pastaImagens, pastaSaida, tipoAtributo, numFolds=5, seed=42):
    
    # garante que a pasta de saída existe para o fold
    os.makedirs(pastaSaida, exist_ok=True)
    
    # agrupa imagens por classe para o fold
    imagensPorClasse = defaultdict(list)
    
    # lista imagens da pasta do tipo de atributo
    todasImagens = [f for f in os.listdir(pastaImagens) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # filtra pelo prefixo da pasta e pega a classe
    for img in todasImagens:
        if not img.startswith(f"{tipoAtributo}_"):
            continue
        
        # extrai classe do nome da imagem
        partes = img.split('_')
        if len(partes) < 3:
            continue
        
        classe = partes[1]  # segunda parte é a classe
        
        # monta caminho original esperado para o fold
        nomeOriginal = '_'.join(partes[2:])
        caminhoOriginal = f"/mnt/data/fabric/dataset_atributos/{tipoAtributo}/{classe}/{nomeOriginal}"
        
        imagensPorClasse[classe].append((classe, caminhoOriginal, img))
    
    # ordena as classes
    classes = sorted(imagensPorClasse.keys())
    
    print(f"Encontradas {len(classes)} classes: {classes}")
    
    # mostra quantidades
    for classe in classes:
        print(f"  {classe}: {len(imagensPorClasse[classe])} imagens")
    
    # so um aviso se alguma classe nao tiver 100 imagens
    for classe, imagens in imagensPorClasse.items():
        if len(imagens) != 100:
            print(f"AVISO: {classe} tem {len(imagens)} imagens (esperado 100)")
    
    # cria cada fold com split 70/30
    for numFold in range(1, numFolds + 1):
        print(f"\nCriando fold {numFold}...")
        
        # seed varia por fold para garantir reprodutibilidade
        random_state_fold = seed + numFold
        
        todasTreino = []
        todasTeste = []
        
        for classe in classes:
            imagens = imagensPorClasse[classe].copy()
            
            # divide 70/30 com train_test_split do sklearn
            treino, teste = train_test_split(
                imagens,
                test_size=0.3,
                random_state=random_state_fold,
                shuffle=True
            )
            
            # adiciona indice e formata: index;classe;caminho
            for idx, (cl, caminho, img) in enumerate(treino):
                todasTreino.append(f"{idx};{cl};{caminho}\n")
            
            for idx, (cl, caminho, img) in enumerate(teste):
                todasTeste.append(f"{idx};{cl};{caminho}\n")
            
            print(f"  {classe}: {len(treino)} treino, {len(teste)} teste")
        
        #aqui salva os arquivos do fold
        arquivoTreino = os.path.join(pastaSaida, f'fold{numFold}-train.txt')
        arquivoTeste = os.path.join(pastaSaida, f'fold{numFold}-test.txt')
        
        with open(arquivoTreino, 'w') as f:
            f.writelines(todasTreino)
        
        with open(arquivoTeste, 'w') as f:
            f.writelines(todasTeste)
        
        print(f"  salvo: {len(todasTreino)} treino, {len(todasTeste)} teste")
    
    print(f"\nfolds criados em: {pastaSaida}")


def main():
    import sys
    
    PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
    pastaDados = os.path.join(PASTA_RAIZ, 'data')
    
    # cor
    print("criando folds para cor")
    criar_folds(
        pastaImagens=os.path.join(pastaDados, 'images', 'color'),
        pastaSaida=os.path.join(pastaDados, 'Protocolo', 'folds_color_70_30', 'folds'),
        tipoAtributo='color',
        numFolds=5,
        seed=42
    )
    
    # textura
    print("criando folds para textura")
    criar_folds(
        pastaImagens=os.path.join(pastaDados, 'images', 'texture'),
        pastaSaida=os.path.join(pastaDados, 'Protocolo', 'folds_texture_70_30', 'folds'),
        tipoAtributo='texture',
        numFolds=5,
        seed=42
    )
    
    print("concluido!")

if __name__ == '__main__':
    main()
