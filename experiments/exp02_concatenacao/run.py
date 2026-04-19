#!/usr/bin/env python3
# experimento 2 - fusao por concatenacao das features early + deep de cada backbone
# a ideia era ver se juntar as duas camadas melhora em relacao a usar so uma
# spoiler: nem sempre melhora, o ibot color caiu bastante com concatenacao
#
# como usar:
#   python run.py                        (roda todos os backbones)
#   python run.py --backbone ibot        (so um backbone)
#   python run.py --backbone resnet50 vmamba

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from torchvision import transforms
import pandas as pd
from datetime import datetime

# caminhos do projeto
PASTA_BASE   = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ   = os.path.dirname(os.path.dirname(PASTA_BASE))
PASTA_EXP1   = os.path.join(PASTA_RAIZ, 'exp1_camada_unica')
PASTA_DADOS  = os.path.join(PASTA_RAIZ, 'data')
PASTA_VMAMBA = os.path.join(PASTA_EXP1, 'vmamba')

sys.path.insert(0, PASTA_RAIZ)
from src.evaluation.retrieval import computar_metricas_retrieval

# melhores camadas de cada backbone (baseado nos resultados do exp01)
# E = melhor camada para cor (early/shallow)
# D = melhor camada para textura (deep)
MELHORES_CAMADAS = {
    'vgg16':   {'color': 'layer1', 'texture': 'layer4'},
    'resnet50':{'color': 'layer2', 'texture': 'layer3'},
    'ibot':    {'color': 'block2', 'texture': 'block9'},  # blocos do ViT, nao layers
    'vmamba':  {'color': 'stage1', 'texture': 'stage2'},
}


def pegar_transformacoes():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def carregar_fold(arquivo_fold, pasta_imagens, atributo):
    caminhos, rotulos = [], []
    with open(arquivo_fold, 'r') as f:
        for linha in f:
            partes = linha.strip().split(';')
            if len(partes) >= 3:
                rotulo = partes[1]
                caminho_original = partes[2]
                nome_arquivo = os.path.basename(caminho_original)
                nome_local = f"{atributo}_{rotulo}_{nome_arquivo}"
                caminho_local = os.path.join(pasta_imagens, nome_local)
                caminhos.append(caminho_local)
                rotulos.append(rotulo)
    return caminhos, rotulos


def carregar_imagens(caminhos, transformacao):
    imagens, indices_validos = [], []
    for i, caminho in enumerate(caminhos):
        if os.path.exists(caminho):
            try:
                img = Image.open(caminho).convert('RGB')
                imagens.append(transformacao(img))
                indices_validos.append(i)
            except Exception:
                pass
    if not imagens:
        return None, []
    return torch.stack(imagens), indices_validos


def carregar_extrator(nome_backbone, dispositivo):
    """Carrega o extrator de features correto para cada backbone."""
    if nome_backbone == 'vgg16':
        pasta = os.path.join(PASTA_EXP1, 'vgg16', 'models')
        sys.path.insert(0, pasta)
        from carregadorVGG import criarExtrator
        return criarExtrator(dispositivo=dispositivo)

    elif nome_backbone == 'resnet50':
        pasta = os.path.join(PASTA_EXP1, 'resnet50', 'models')
        sys.path.insert(0, pasta)
        from carregadorResNet import criarExtrator
        return criarExtrator(dispositivo=dispositivo)

    elif nome_backbone == 'ibot':
        pasta = os.path.join(PASTA_EXP1, 'ibot', 'models')
        sys.path.insert(0, pasta)
        from carregadorIBot import criarExtrator
        return criarExtrator(modelo='vit_small', dispositivo=dispositivo)

    elif nome_backbone == 'vmamba':
        sys.path.insert(0, PASTA_VMAMBA)
        spec = importlib.util.spec_from_file_location(
            'carregadorVMamba',
            os.path.join(PASTA_VMAMBA, 'models', 'carregadorVMamba.py')
        )
        modulo = importlib.util.module_from_spec(spec)
        sys.modules['carregadorVMamba'] = modulo
        spec.loader.exec_module(modulo)
        return modulo.criarExtrator(dispositivo=dispositivo)

    else:
        raise ValueError(f"backbone desconhecido: {nome_backbone}")


def extrair_features_lote(extrator, imagens, nome_backbone, dispositivo, tamanho_lote=32):
    """Extrai features em lotes usando o metodo correto de cada backbone.

    - iBOT: usa CLS token (consistente com o exp03 MoE)
    - demais: usa extrairFeatures com GAP
    """
    todas = {}
    n = len(imagens)

    for i in range(0, n, tamanho_lote):
        lote = imagens[i:i + tamanho_lote].to(dispositivo)

        if nome_backbone == 'ibot':
            feats = extrator.extrairFeaturesComCLS(lote)
        else:
            feats = extrator.extrairFeatures(lote, aplicarGAP=True)

        for nome, feat in feats.items():
            # os extratores ja retornam tensores CPU; converte pra numpy
            feat_np = feat.cpu().numpy() if hasattr(feat, 'cpu') else np.array(feat)
            if nome not in todas:
                todas[nome] = []
            todas[nome].append(feat_np)

    for nome in todas:
        todas[nome] = np.concatenate(todas[nome], axis=0)

    return todas


def media_retrieval(lista_dicts):
    """Recebe lista de dicts {map_at_10, r_at_1, r_at_5} e retorna a media de cada metrica."""
    keys = ['map_at_10', 'r_at_1', 'r_at_5']
    return {k: np.mean([d[k] for d in lista_dicts]) for k in keys}


def avaliar_knn(X_treino, y_treino, X_teste, y_teste, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_treino, y_treino)
    y_pred = knn.predict(X_teste)
    acc = accuracy_score(y_teste, y_pred) * 100
    f1  = f1_score(y_teste, y_pred, average='macro') * 100
    return acc, f1


def rodar_experimento(nome_backbone, dispositivo='cuda'):
    print(f"\n{'='*60}")
    print(f"BACKBONE: {nome_backbone.upper()}")
    print(f"{'='*60}")

    extrator = carregar_extrator(nome_backbone, dispositivo)

    config       = MELHORES_CAMADAS[nome_backbone]
    camada_early = config['color']    # E = camada rasa (cor)
    camada_deep  = config['texture']  # D = camada profunda (textura)

    print(f"Camada E (cor):     {camada_early}")
    print(f"Camada D (textura): {camada_deep}")

    transformacao = pegar_transformacoes()
    resultados = []

    for atributo in ['color', 'texture']:
        print(f"\n--- Atributo: {atributo.upper()} ---")

        pasta_folds   = os.path.join(PASTA_DADOS, 'protocols', f'folds_{atributo}_70_30', 'folds')
        pasta_imagens = os.path.join(PASTA_DADOS, 'images', atributo)

        acc_E, acc_D, acc_Z = [], [], []
        f1_E,  f1_D,  f1_Z  = [], [], []
        ret_E, ret_D, ret_Z = [], [], []

        for num_fold in range(1, 6):
            print(f"  Fold {num_fold}/5...", end=" ", flush=True)

            caminhos_treino, rotulos_treino = carregar_fold(
                os.path.join(pasta_folds, f'fold{num_fold}-train.txt'), pasta_imagens, atributo)
            caminhos_teste, rotulos_teste = carregar_fold(
                os.path.join(pasta_folds, f'fold{num_fold}-test.txt'), pasta_imagens, atributo)

            imagens_treino, idx_treino = carregar_imagens(caminhos_treino, transformacao)
            imagens_teste,  idx_teste  = carregar_imagens(caminhos_teste,  transformacao)

            if imagens_treino is None or imagens_teste is None:
                print("ERRO: imagens nao encontradas")
                continue

            rotulos_treino_val = [rotulos_treino[i] for i in idx_treino]
            rotulos_teste_val  = [rotulos_teste[i]  for i in idx_teste]

            feats_treino = extrair_features_lote(extrator, imagens_treino, nome_backbone, dispositivo)
            feats_teste  = extrair_features_lote(extrator, imagens_teste,  nome_backbone, dispositivo)

            E_tr = feats_treino[camada_early]
            D_tr = feats_treino[camada_deep]
            E_te = feats_teste[camada_early]
            D_te = feats_teste[camada_deep]

            # fusao por concatenacao: Z = [E || D]
            Z_tr = np.concatenate([E_tr, D_tr], axis=1)
            Z_te = np.concatenate([E_te, D_te], axis=1)

            a, f = avaliar_knn(E_tr, rotulos_treino_val, E_te, rotulos_teste_val)
            acc_E.append(a); f1_E.append(f)

            a, f = avaliar_knn(D_tr, rotulos_treino_val, D_te, rotulos_teste_val)
            acc_D.append(a); f1_D.append(f)

            a, f = avaliar_knn(Z_tr, rotulos_treino_val, Z_te, rotulos_teste_val)
            acc_Z.append(a); f1_Z.append(f)

            ret_E.append(computar_metricas_retrieval(E_tr, rotulos_treino_val, E_te, rotulos_teste_val))
            ret_D.append(computar_metricas_retrieval(D_tr, rotulos_treino_val, D_te, rotulos_teste_val))
            ret_Z.append(computar_metricas_retrieval(Z_tr, rotulos_treino_val, Z_te, rotulos_teste_val))

            print(f"E={acc_E[-1]:.1f}%  D={acc_D[-1]:.1f}%  Z={acc_Z[-1]:.1f}%")

        mr_E = media_retrieval(ret_E)
        mr_D = media_retrieval(ret_D)
        mr_Z = media_retrieval(ret_Z)

        resultado = {
            'backbone':    nome_backbone,
            'atributo':    atributo,
            'camada_E':    camada_early,
            'acc_E_media': np.mean(acc_E), 'acc_E_std': np.std(acc_E), 'f1_E_media': np.mean(f1_E),
            'map10_E': mr_E['map_at_10'], 'r1_E': mr_E['r_at_1'], 'r5_E': mr_E['r_at_5'],
            'camada_D':    camada_deep,
            'acc_D_media': np.mean(acc_D), 'acc_D_std': np.std(acc_D), 'f1_D_media': np.mean(f1_D),
            'map10_D': mr_D['map_at_10'], 'r1_D': mr_D['r_at_1'], 'r5_D': mr_D['r_at_5'],
            'acc_Z_media': np.mean(acc_Z), 'acc_Z_std': np.std(acc_Z), 'f1_Z_media': np.mean(f1_Z),
            'map10_Z': mr_Z['map_at_10'], 'r1_Z': mr_Z['r_at_1'], 'r5_Z': mr_Z['r_at_5'],
        }
        resultado['melhora'] = (resultado['acc_Z_media']
                                - max(resultado['acc_E_media'], resultado['acc_D_media']))
        resultados.append(resultado)

        print(f"\n  Resultado {atributo}:")
        print(f"    E ({camada_early}): acc={resultado['acc_E_media']:.2f}%  mAP@10={mr_E['map_at_10']:.2f}%  R@1={mr_E['r_at_1']:.2f}%  R@5={mr_E['r_at_5']:.2f}%")
        print(f"    D ({camada_deep}):  acc={resultado['acc_D_media']:.2f}%  mAP@10={mr_D['map_at_10']:.2f}%  R@1={mr_D['r_at_1']:.2f}%  R@5={mr_D['r_at_5']:.2f}%")
        print(f"    Z (fusao):         acc={resultado['acc_Z_media']:.2f}%  mAP@10={mr_Z['map_at_10']:.2f}%  R@1={mr_Z['r_at_1']:.2f}%  R@5={mr_Z['r_at_5']:.2f}%")
        print(f"    Melhora acc:       {resultado['melhora']:+.2f}%")

    return resultados


def main():
    parser = argparse.ArgumentParser(description='exp02 - fusao por concatenacao early+deep')
    parser.add_argument('--backbone', nargs='+',
                        choices=['vgg16', 'resnet50', 'ibot', 'vmamba'],
                        default=['vgg16', 'resnet50', 'ibot', 'vmamba'],
                        help='backbone(s) a executar (padrao: todos)')
    args = parser.parse_args()

    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {dispositivo}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")

    todos = []
    for nome in args.backbone:
        try:
            todos.extend(rodar_experimento(nome, dispositivo))
        except Exception as e:
            print(f"ERRO com {nome}: {e}")
            import traceback; traceback.print_exc()

    if not todos:
        print("nenhum resultado gerado")
        return

    df = pd.DataFrame(todos)
    saida = os.path.join(PASTA_BASE, 'results', 'fusao.csv')
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    df.to_csv(saida, index=False)
    print(f"\nResultados salvos em: {saida}")

    print(f"\n{'='*90}")
    print("RESUMO — EXP02: FUSAO EARLY-DEEP POR CONCATENACAO (70/30, 5-fold)")
    print(f"{'='*90}")
    header = f"{'Backbone':<12} {'Atrib.':<10} {'Método':<14} {'Acc(%)':>7} {'mAP@10':>8} {'R@1':>7} {'R@5':>7}"
    print(header)
    print("-" * 70)
    for r in todos:
        for metodo, suffix in [('Early (E)', '_E'), ('Deep (D)', '_D'), ('Concat (Z)', '_Z')]:
            print(f"{r['backbone']:<12} {r['atributo']:<10} {metodo:<14} "
                  f"{r[f'acc{suffix}_media']:>7.2f} "
                  f"{r[f'map10{suffix}']:>8.2f} "
                  f"{r[f'r1{suffix}']:>7.2f} "
                  f"{r[f'r5{suffix}']:>7.2f}")
        print()

    print(f"\nFim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
