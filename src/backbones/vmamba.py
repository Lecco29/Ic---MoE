# carregador do vmamba usando modelo do huggingface
# esse arquivo carrega o modelo e extrai features das camadas intermediarias

import os
import sys
import torch

# adiciona os caminhos necessarios
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, 'vmamba_repo', 'kernels', 'selective_scan'))


class ExtratorVMamba:
    # essa classe carrega o vmamba e extrai features de cada estagio
    # usa o modelo do huggingface que ja vem com os pesos treinados
    
    def __init__(self, dispositivo='auto'):
        # dispositivo = 'cuda', 'cpu' ou 'auto' (detecta automatico)
        
        # configura o dispositivo
        if dispositivo == 'auto':
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo
        
        # dimensoes de cada estagio do vmamba tiny
        self.dimEstagios = {
            'stage1': 96,
            'stage2': 192,
            'stage3': 384,
            'stage4': 768
        }
        
        # guarda as features capturadas pelos hooks
        self.features = {}
        self.hooks = []
        
        # carrega o modelo
        self.carregarModelo()
    
    def carregarModelo(self):
        # carrega o modelo vmamba do huggingface
        from modeling_vmamba import VMambaForImageClassification
        from configuration_vmamba import VMambaConfig
        from safetensors.torch import load_file
        
        # cria o modelo
        config = VMambaConfig()
        self.modelo = VMambaForImageClassification(config)
        
        # carrega os pesos pre treinados
        caminhoPesos = os.path.join(RAIZ, 'model.safetensors')
        if os.path.exists(caminhoPesos):
            pesos = load_file(caminhoPesos)
            self.modelo.load_state_dict(pesos)
            print("[VMamba] Pesos carregados!")
        else:
            print("[VMamba] AVISO: pesos nao encontrados, usando pesos aleatorios")
        
        self.modelo = self.modelo.to(self.dispositivo)
        self.modelo.eval()
        
        # registra os hooks para capturar features
        self.registrarHooks()
        
        print(f"[VMamba] Dispositivo: {self.dispositivo}")
        print(f"[VMamba] Dimensoes: {self.dimEstagios}")
    
    def registrarHooks(self):
        # registra hooks em cada estagio do modelo
        # hooks sao funcoes que capturam a saida de cada camada
        
        def criarHook(nome):
            def hook(modulo, entrada, saida):
                self.features[nome] = saida
            return hook
        
        # limpa hooks antigos
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # registra hooks nos layers do vmamba
        vmamba = self.modelo.vmamba
        if hasattr(vmamba, 'layers'):
            for indice, camada in enumerate(vmamba.layers):
                hook = camada.register_forward_hook(criarHook(f'stage{indice+1}'))
                self.hooks.append(hook)
    
    def extrairFeatures(self, entrada, aplicarGAP=True):
        # extrai features de todos os estagios
        # entrada = tensor de entrada [B, C, H, W]
        # aplicarGAP = se True, faz Global Average Pooling
        # retorna dicionario com features de cada estagio
        
        self.features = {}
        
        with torch.no_grad():
            _ = self.modelo(entrada)
        
        resultado = {}
        for nome, feature in self.features.items():
            if aplicarGAP:
                # global average pooling: transforma [B,C,H,W] em [B,C]
                if len(feature.shape) == 4:
                    feature = feature.mean(dim=[2, 3])
                elif len(feature.shape) == 3:
                    feature = feature.mean(dim=1)
            resultado[nome] = feature.cpu()
        
        return resultado
    
    def pegarDimensoes(self):
        # retorna as dimensoes de cada estagio
        return self.dimEstagios.copy()
    
    # propriedade para manter compatibilidade
    @property
    def stage_dims(self):
        return self.dimEstagios


def criarExtrator(dispositivo='auto'):
    # funcao para criar o extrator de features
    return ExtratorVMamba(dispositivo=dispositivo)


