# extracao de features da resnet50 em batches

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.carregadorResNet import criarExtrator


class DatasetImagens(Dataset):

    def __init__(self, caminhos, transform=None):
        self.caminhos = caminhos
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        caminho = self.caminhos[idx]
        imagem = Image.open(caminho).convert('RGB')
        if self.transform:
            imagem = self.transform(imagem)
        return imagem, caminho


class ExtratorFeatures:

    def __init__(self, dispositivo='auto', tamanhoBatch=32):
        self.extrator = criarExtrator(dispositivo)
        self.tamanhoBatch = tamanhoBatch
        self.dispositivo = self.extrator.dispositivo
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extrairDeLista(self, caminhos):
        dataset = DatasetImagens(caminhos, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.tamanhoBatch,
                                shuffle=False, num_workers=4, pin_memory=True)
        todasFeatures = {nome: [] for nome in self.extrator.dimCamadas.keys()}
        todosCaminhos = []

        print(f"[extracao] processando {len(caminhos)} imagens...")
        for batchImagens, batchCaminhos in tqdm(dataloader, desc="extraindo features"):
            batchImagens = batchImagens.to(self.dispositivo)
            featuresBatch = self.extrator.extrairFeatures(batchImagens, aplicarGAP=True)
            for nomeCamada, feat in featuresBatch.items():
                todasFeatures[nomeCamada].append(feat)
            todosCaminhos.extend(batchCaminhos)

        for nomeCamada in todasFeatures:
            todasFeatures[nomeCamada] = torch.cat(todasFeatures[nomeCamada], dim=0)

        return todasFeatures, todosCaminhos

    def pegarDimensoes(self):
        return self.extrator.pegarDimensoes()


def criarExtratorFeatures(dispositivo='auto', tamanhoBatch=32):
    return ExtratorFeatures(dispositivo=dispositivo, tamanhoBatch=tamanhoBatch)
