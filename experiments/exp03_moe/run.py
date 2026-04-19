import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# caminhos do projeto
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(os.path.dirname(PASTA_BASE))
PASTA_DADOS = os.path.join(PASTA_RAIZ, 'data')

sys.path.insert(0, PASTA_BASE)
sys.path.insert(0, PASTA_RAIZ)
from models.moe import criar_modelo_moe
from src.evaluation.retrieval import computar_metricas_retrieval


# dataset simples que carrega imagem e retorna o label como inteiro
class ConjuntoDados(Dataset):

    def __init__(self, caminhos, rotulos, transformacao=None):
        self.caminhos = caminhos
        self.rotulos = rotulos
        self.transformacao = transformacao

        # mapeia os nomes de classe pra indices numericos
        classes = sorted(set(rotulos))
        self.classe_para_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        img = Image.open(self.caminhos[idx]).convert('RGB')
        if self.transformacao:
            img = self.transformacao(img)
        rotulo_idx = self.classe_para_idx[self.rotulos[idx]]
        return img, torch.tensor(rotulo_idx, dtype=torch.long)


def pegar_transformacoes(treino=True):
    # no treino usa augmentation basica pra nao overfitar
    if treino:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    # na avaliacao so redimensiona e normaliza
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def carregar_fold(arquivo_fold, pasta_imagens, atributo):
    # le o arquivo txt do fold e monta os caminhos das imagens locais
    caminhos, rotulos = [], []
    with open(arquivo_fold) as f:
        for linha in f:
            partes = linha.strip().split(';')
            if len(partes) < 3:
                continue
            rotulo = partes[1]
            original = partes[2]
            nome_arquivo = f"{atributo}_{original.split('/')[-2]}_{original.split('/')[-1]}"
            caminho = os.path.join(pasta_imagens, nome_arquivo)
            if os.path.exists(caminho):
                caminhos.append(caminho)
                rotulos.append(rotulo)
    return caminhos, rotulos


def treinar_epoca(modelo, cabeca, loader, otimizador, criterio, dispositivo):
    modelo.train()
    cabeca.train()
    loss_total, acertos, total = 0.0, 0, 0

    for imgs, rotulos in loader:
        imgs, rotulos = imgs.to(dispositivo), rotulos.to(dispositivo)
        otimizador.zero_grad()
        z, _, _ = modelo(imgs)
        logits = cabeca(z)
        loss = criterio(logits, rotulos)
        loss.backward()
        otimizador.step()
        loss_total += loss.item()
        acertos += (logits.detach().argmax(1) == rotulos).sum().item()
        total += rotulos.size(0)

    return loss_total / len(loader), 100.0 * acertos / total


@torch.no_grad()
def extrair_embeddings(modelo, loader, dispositivo):
    # passa todas as imagens pelo modelo e coleta os vetores z
    modelo.eval()
    vetores, rotulos = [], []
    for imgs, rots in loader:
        z, _, _ = modelo(imgs.to(dispositivo))
        vetores.append(z.cpu().numpy())
        rotulos.extend(rots.numpy())
    return np.vstack(vetores), np.array(rotulos)


def rodar_experimento(nome_backbone, atributo, dispositivo,
                      epocas=30, d=256, tamanho_lote=32,
                      lr=1e-4, k_knn=5, so_avaliar=False):

    print(f"\n{'='*60}")
    print(f"MoE | Backbone: {nome_backbone.upper()} | Atributo: {atributo.upper()}")
    print(f"Epocas: {epocas} | d={d} | lr={lr} | KNN k={k_knn}")
    print(f"{'='*60}")

    pasta_imagens = os.path.join(PASTA_DADOS, 'images', atributo)
    pasta_folds = os.path.join(PASTA_DADOS, 'protocols', f'folds_{atributo}_70_30', 'folds')

    if not os.path.isdir(pasta_imagens):
        raise FileNotFoundError(f"pasta de imagens nao encontrada: {pasta_imagens}")
    if not os.path.isdir(pasta_folds):
        raise FileNotFoundError(f"pasta de folds nao encontrada: {pasta_folds}")

    resultados = []
    metricas_retrieval = []

    for fold in range(1, 6):
        print(f"\n--- Fold {fold}/5 ---")

        caminhos_treino, rotulos_treino = carregar_fold(
            os.path.join(pasta_folds, f'fold{fold}-train.txt'), pasta_imagens, atributo)
        caminhos_teste, rotulos_teste = carregar_fold(
            os.path.join(pasta_folds, f'fold{fold}-test.txt'), pasta_imagens, atributo)

        if not caminhos_treino:
            print(f"  sem imagens no fold {fold}, pulando")
            continue
        print(f"  treino: {len(caminhos_treino)} | teste: {len(caminhos_teste)}")

        ds_treino = ConjuntoDados(caminhos_treino, rotulos_treino, pegar_transformacoes(treino=True))
        ds_teste = ConjuntoDados(caminhos_teste, rotulos_teste, pegar_transformacoes(treino=False))
        # esse aqui e sem augmentation, so pra extrair os embeddings depois do treino
        ds_treino_eval = ConjuntoDados(caminhos_treino, rotulos_treino, pegar_transformacoes(treino=False))

        loader_treino = DataLoader(ds_treino, batch_size=tamanho_lote, shuffle=True,
                                   num_workers=4, pin_memory=True)
        loader_teste = DataLoader(ds_teste, batch_size=tamanho_lote, shuffle=False,
                                  num_workers=4, pin_memory=True)
        loader_treino_eval = DataLoader(ds_treino_eval, batch_size=tamanho_lote, shuffle=False,
                                        num_workers=4, pin_memory=True)

        num_classes = ds_treino.num_classes
        print(f"  classes: {num_classes}")

        modelo = criar_modelo_moe(nome_backbone, d=d, pre_treinado=True).to(dispositivo)

        if not so_avaliar:
            # cabeca linear so pra supervisionar o treino, nao e usada na avaliacao final
            cabeca = nn.Linear(d, num_classes).to(dispositivo)
            criterio = nn.CrossEntropyLoss()

            # backbone ja congelado, treina so hE + hD + router + cabeca
            parametros_treinaveis = (list(modelo.hE.parameters()) +
                                     list(modelo.hD.parameters()) +
                                     list(modelo.router.parameters()) +
                                     list(cabeca.parameters()))
            otimizador = optim.AdamW(parametros_treinaveis, lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)

            melhor_loss = float('inf')
            for epoca in range(epocas):
                loss, acc = treinar_epoca(modelo, cabeca, loader_treino,
                                          otimizador, criterio, dispositivo)
                scheduler.step()
                if loss < melhor_loss:
                    melhor_loss = loss
                if (epoca + 1) % 10 == 0 or epoca == 0:
                    print(f"  epoca {epoca+1:3d}/{epocas}: loss={loss:.4f} acc={acc:.2f}%")
        else:
            print("  [so_avaliar] pulando treino")

        # extrai os embeddings z = alpha*fE + beta*fD de treino e teste
        print("  extraindo embeddings...")
        z_treino, y_treino = extrair_embeddings(modelo, loader_treino_eval, dispositivo)
        z_teste, y_teste = extrair_embeddings(modelo, loader_teste, dispositivo)

        # normaliza e avalia com KNN
        normalizador = StandardScaler()
        z_treino_norm = normalizador.fit_transform(z_treino)
        z_teste_norm = normalizador.transform(z_teste)

        knn = KNeighborsClassifier(n_neighbors=k_knn, n_jobs=-1)
        knn.fit(z_treino_norm, y_treino)
        predicoes = knn.predict(z_teste_norm)

        acc_knn = accuracy_score(y_teste, predicoes) * 100
        f1_knn = f1_score(y_teste, predicoes, average='weighted') * 100

        ret = computar_metricas_retrieval(z_treino_norm, y_treino, z_teste_norm, y_teste)
        metricas_retrieval.append(ret)

        print(f"  KNN acc: {acc_knn:.2f}%  F1: {f1_knn:.2f}%")
        print(f"  CBIR  mAP@10: {ret['map_at_10']:.2f}%  R@1: {ret['r_at_1']:.2f}%  R@5: {ret['r_at_5']:.2f}%")
        resultados.append({'fold': fold, 'acuracia': acc_knn, 'f1': f1_knn,
                           'map_at_10': ret['map_at_10'], 'r_at_1': ret['r_at_1'], 'r_at_5': ret['r_at_5']})

    if not resultados:
        print("nenhum fold processado")
        return None

    accs = [r['acuracia'] for r in resultados]
    f1s = [r['f1'] for r in resultados]
    media_acc = np.mean(accs)
    std_acc = np.std(accs)
    media_f1 = np.mean(f1s)

    media_map  = np.mean([r['map_at_10'] for r in resultados])
    media_r1   = np.mean([r['r_at_1']    for r in resultados])
    media_r5   = np.mean([r['r_at_5']    for r in resultados])

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL - {nome_backbone.upper()} {atributo.upper()}")
    print(f"  Acuracia: {media_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  F1:       {media_f1:.2f}%")
    print(f"  mAP@10:   {media_map:.2f}%")
    print(f"  R@1:      {media_r1:.2f}%")
    print(f"  R@5:      {media_r5:.2f}%")
    print(f"{'='*60}")

    df = pd.DataFrame(resultados)
    df['acuracia_media'] = media_acc
    df['acuracia_std'] = std_acc
    df['f1_media'] = media_f1
    df['map_at_10_media'] = media_map
    df['r_at_1_media'] = media_r1
    df['r_at_5_media'] = media_r5
    arquivo_csv = os.path.join(PASTA_BASE, 'results', f'{nome_backbone}_{atributo}.csv')
    os.makedirs(os.path.dirname(arquivo_csv), exist_ok=True)
    df.to_csv(arquivo_csv, index=False)
    print(f"salvo em: {arquivo_csv}")

    return {
        'acuracia_media': media_acc, 'acuracia_std': std_acc, 'f1_media': media_f1,
        'map_at_10': media_map, 'r_at_1': media_r1, 'r_at_5': media_r5,
    }


def main():
    parser = argparse.ArgumentParser(description='treina e avalia o modelo MoE')
    parser.add_argument('--backbone', type=str, default='ibot',
                        choices=['ibot', 'resnet50', 'vmamba', 'vgg16'])
    parser.add_argument('--atributo', type=str, default='color',
                        choices=['color', 'texture', 'both'])
    parser.add_argument('--epocas', type=int, default=30)
    parser.add_argument('--d', type=int, default=256,
                        help='dimensao das projecoes hE e hD')
    parser.add_argument('--lote', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=5,
                        help='k do KNN')
    parser.add_argument('--so_avaliar', action='store_true',
                        help='pula o treino, util pra debug')
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"dispositivo: {dispositivo}")
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")

    atributos = ['color', 'texture'] if args.atributo == 'both' else [args.atributo]

    todos = {}
    for atributo in atributos:
        res = rodar_experimento(
            nome_backbone=args.backbone,
            atributo=atributo,
            dispositivo=dispositivo,
            epocas=args.epocas,
            d=args.d,
            tamanho_lote=args.lote,
            lr=args.lr,
            k_knn=args.k,
            so_avaliar=args.so_avaliar,
        )
        if res:
            todos[atributo] = res

    print(f"\nfim: {datetime.now().strftime('%H:%M:%S')}")

    if len(todos) > 1:
        print("\n=== resumo ===")
        print(f"  {'Atributo':<10} {'Acc(%)':>8} {'mAP@10':>8} {'R@1':>7} {'R@5':>7}")
        print("  " + "-" * 45)
        for atributo, res in todos.items():
            print(f"  {atributo:<10} {res['acuracia_media']:>8.2f} "
                  f"{res['map_at_10']:>8.2f} {res['r_at_1']:>7.2f} {res['r_at_5']:>7.2f}")


if __name__ == '__main__':
    main()
