# Codigo para gerar os graficos de desempenho por camada

import matplotlib.pyplot as plt
import numpy as np
import os

# aumenta as letras do matplotlib (para os png ficarem legiveis no artigo)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})

# Dados do experimento 1

# VGG-16
vgg16_camadas = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']
vgg16_cor = [95.13, 90.27, 85.47, 71.80, 55.07]
vgg16_cor_std = [0.81, 1.27, 2.33, 2.55, 2.64]
vgg16_textura = [70.73, 86.60, 93.20, 94.73, 89.40]
vgg16_textura_std = [2.25, 1.57, 1.48, 0.74, 1.64]

# ResNet-50
resnet_camadas = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
resnet_cor = [95.47, 95.87, 88.27, 59.20]
resnet_cor_std = [0.78, 0.50, 1.12, 3.11]
resnet_textura = [83.40, 90.67, 93.27, 86.33]
resnet_textura_std = [1.36, 1.65, 0.25, 0.92]

# iBOT
ibot_camadas = ['Block ' + str(i) for i in range(12)]
ibot_cor = [94.53, 95.40, 95.87, 95.20, 92.13, 88.73, 84.93, 84.07, 76.00, 69.93, 66.13, 62.80]
ibot_cor_std = [1.13, 1.27, 1.05, 0.83, 0.27, 0.77, 1.24, 1.18, 0.76, 1.97, 3.28, 1.73]
ibot_textura = [52.20, 69.87, 79.93, 87.87, 90.53, 91.00, 92.20, 95.00, 96.87, 97.20, 96.27, 96.00]
ibot_textura_std = [2.69, 2.28, 2.61, 1.44, 1.33, 2.02, 1.85, 0.99, 0.54, 0.54, 0.53, 0.87]

# VMamba
vmamba_camadas = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
vmamba_cor = [94.41, 93.41, 78.34, 54.14]
vmamba_cor_std = [0.52, 0.47, 3.75, 2.39]
vmamba_textura = [93.86, 95.76, 95.76, 86.52]
vmamba_textura_std = [1.07, 0.84, 0.75, 2.57]


# GERANDO OS GRAFICOS

pasta = os.path.dirname(os.path.abspath(__file__))

# primeira geracao de graficos todos os backbones em grid 2x2 ----------

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# helper simples pra marcar a melhor camada de cor (E) e textura (D)
def marcar_melhores(ax, x_vals, y_cor, y_textura, idx_cor, idx_textura, label_cor, label_textura):
    # marca o melhor ponto de cor (E)
    ax.scatter([x_vals[idx_cor]], [y_cor[idx_cor]], s=220, c='#d62728', edgecolors='black', linewidths=2, zorder=5)
    ax.text(x_vals[idx_cor] + 0.05, y_cor[idx_cor] + 3, f"E={label_cor}", fontsize=13, color='#8c564b')
    # marca o melhor ponto de textura (D)
    ax.scatter([x_vals[idx_textura]], [y_textura[idx_textura]], s=220, c='#1f77b4', marker='s',
               edgecolors='black', linewidths=2, zorder=5)
    ax.text(x_vals[idx_textura] + 0.05, y_textura[idx_textura] - 6, f"D={label_textura}", fontsize=13, color='#1f77b4')

# VGG-16
ax = axs[0, 0]
x = range(1, len(vgg16_camadas) + 1)
ax.plot(x, vgg16_cor, 'o-', color='red', label='Color', linewidth=2)
ax.fill_between(x, np.array(vgg16_cor) - np.array(vgg16_cor_std),
                np.array(vgg16_cor) + np.array(vgg16_cor_std), color='red', alpha=0.2)
ax.plot(x, vgg16_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.fill_between(x, np.array(vgg16_textura) - np.array(vgg16_textura_std),
                np.array(vgg16_textura) + np.array(vgg16_textura_std), color='blue', alpha=0.2)
ax.set_title('VGG-16 (CNN)', fontweight='bold')
ax.set_xlabel('Layer Depth')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(vgg16_camadas, rotation=45, ha='right')
ax.set_ylim(45, 100)
ax.legend()
ax.grid(True, alpha=0.3)
marcar_melhores(ax, list(x), vgg16_cor, vgg16_textura, idx_cor=0, idx_textura=3, label_cor='Layer 1', label_textura='Layer 4')

# ResNet-50
ax = axs[0, 1]
x = range(1, len(resnet_camadas) + 1)
ax.plot(x, resnet_cor, 'o-', color='red', label='Color', linewidth=2)
ax.fill_between(x, np.array(resnet_cor) - np.array(resnet_cor_std),
                np.array(resnet_cor) + np.array(resnet_cor_std), color='red', alpha=0.2)
ax.plot(x, resnet_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.fill_between(x, np.array(resnet_textura) - np.array(resnet_textura_std),
                np.array(resnet_textura) + np.array(resnet_textura_std), color='blue', alpha=0.2)
ax.set_title('ResNet-50 (CNN)', fontweight='bold')
ax.set_xlabel('Layer Depth')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(resnet_camadas, rotation=45, ha='right')
ax.set_ylim(45, 100)
ax.legend()
ax.grid(True, alpha=0.3)
marcar_melhores(ax, list(x), resnet_cor, resnet_textura, idx_cor=1, idx_textura=2, label_cor='Layer 2', label_textura='Layer 3')

# iBOT
ax = axs[1, 0]
x = range(1, len(ibot_camadas) + 1)
ax.plot(x, ibot_cor, 'o-', color='red', label='Color', linewidth=2)
ax.fill_between(x, np.array(ibot_cor) - np.array(ibot_cor_std),
                np.array(ibot_cor) + np.array(ibot_cor_std), color='red', alpha=0.2)
ax.plot(x, ibot_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.fill_between(x, np.array(ibot_textura) - np.array(ibot_textura_std),
                np.array(ibot_textura) + np.array(ibot_textura_std), color='blue', alpha=0.2)
ax.set_title('iBOT (Vision Transformer)', fontweight='bold')
ax.set_xlabel('Layer Depth')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(ibot_camadas, rotation=45, ha='right', fontsize=8)
ax.set_ylim(45, 100)
ax.legend()
ax.grid(True, alpha=0.3)
marcar_melhores(ax, list(x), ibot_cor, ibot_textura, idx_cor=2, idx_textura=9, label_cor='Block 2', label_textura='Block 9')

# VMamba
ax = axs[1, 1]
x = range(1, len(vmamba_camadas) + 1)
ax.plot(x, vmamba_cor, 'o-', color='red', label='Color', linewidth=2)
ax.fill_between(x, np.array(vmamba_cor) - np.array(vmamba_cor_std),
                np.array(vmamba_cor) + np.array(vmamba_cor_std), color='red', alpha=0.2)
ax.plot(x, vmamba_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.fill_between(x, np.array(vmamba_textura) - np.array(vmamba_textura_std),
                np.array(vmamba_textura) + np.array(vmamba_textura_std), color='blue', alpha=0.2)
ax.set_title('VMamba (State Space Model)', fontweight='bold')
ax.set_xlabel('Layer Depth')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(vmamba_camadas, rotation=45, ha='right')
ax.set_ylim(45, 100)
ax.legend()
ax.grid(True, alpha=0.3)
marcar_melhores(ax, list(x), vmamba_cor, vmamba_textura, idx_cor=0, idx_textura=1, label_cor='Stage 1', label_textura='Stage 2')

plt.suptitle('Performance × Depth: Color and Texture Classification', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig(pasta + '/curvas_desempenho_profundidade.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curvas_desempenho_profundidade.png')


# segunda geracao de graficos comparacao entre arquiteturas 

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Normalizando a profundidade pra comparar (0 a 1)
vgg_x = np.linspace(0, 1, len(vgg16_cor))
resnet_x = np.linspace(0, 1, len(resnet_cor))
ibot_x = np.linspace(0, 1, len(ibot_cor))
vmamba_x = np.linspace(0, 1, len(vmamba_cor))

# Grafico de cor
ax = axs[0]
ax.plot(vgg_x, vgg16_cor, 'o-', label='VGG-16', linewidth=2)
ax.plot(resnet_x, resnet_cor, 's-', label='ResNet-50', linewidth=2)
ax.plot(ibot_x, ibot_cor, '^-', label='iBOT', linewidth=2)
ax.plot(vmamba_x, vmamba_cor, 'D-', label='VMamba', linewidth=2)
ax.set_xlabel('Normalized Depth (0=early, 1=deep)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Color Classification', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 100)

# Grafico de textura
ax = axs[1]
ax.plot(vgg_x, vgg16_textura, 'o-', label='VGG-16', linewidth=2)
ax.plot(resnet_x, resnet_textura, 's-', label='ResNet-50', linewidth=2)
ax.plot(ibot_x, ibot_textura, '^-', label='iBOT', linewidth=2)
ax.plot(vmamba_x, vmamba_textura, 'D-', label='VMamba', linewidth=2)
ax.set_xlabel('Normalized Depth (0=early, 1=deep)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Texture Classification', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 100)

plt.suptitle('Comparison Across Architectures', fontsize=26, fontweight='bold')
plt.tight_layout()
plt.savefig(pasta + '/curvas_comparacao_arquiteturas.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curvas_comparacao_arquiteturas.png')


# terceira geracao de graficos individuas

# VGG-16 individual
fig, ax = plt.subplots(figsize=(8, 5))
x = range(1, len(vgg16_camadas) + 1)
ax.plot(x, vgg16_cor, 'o-', color='red', label='Color', linewidth=2)
ax.plot(x, vgg16_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.set_title('VGG-16 - Desempenho x Profundidade', fontweight='bold')
ax.set_xlabel('Camada')
ax.set_ylabel('Acuracia (%)')
ax.set_xticks(x)
ax.set_xticklabels(vgg16_camadas, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pasta + '/curva_vgg16.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curva_vgg16.png')

# ResNet-50 individual
fig, ax = plt.subplots(figsize=(8, 5))
x = range(1, len(resnet_camadas) + 1)
ax.plot(x, resnet_cor, 'o-', color='red', label='Color', linewidth=2)
ax.plot(x, resnet_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.set_title('ResNet-50 - Desempenho x Profundidade', fontweight='bold')
ax.set_xlabel('Camada')
ax.set_ylabel('Acuracia (%)')
ax.set_xticks(x)
ax.set_xticklabels(resnet_camadas, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pasta + '/curva_resnet50.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curva_resnet50.png')

# iBOT individual
fig, ax = plt.subplots(figsize=(8, 5))
x = range(1, len(ibot_camadas) + 1)
ax.plot(x, ibot_cor, 'o-', color='red', label='Color', linewidth=2)
ax.plot(x, ibot_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.set_title('iBOT - Desempenho x Profundidade', fontweight='bold')
ax.set_xlabel('Camada')
ax.set_ylabel('Acuracia (%)')
ax.set_xticks(x)
ax.set_xticklabels(ibot_camadas, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pasta + '/curva_ibot.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curva_ibot.png')

# VMamba individual
fig, ax = plt.subplots(figsize=(8, 5))
x = range(1, len(vmamba_camadas) + 1)
ax.plot(x, vmamba_cor, 'o-', color='red', label='Color', linewidth=2)
ax.plot(x, vmamba_textura, 's-', color='blue', label='Texture', linewidth=2)
ax.set_title('VMamba - Desempenho x Profundidade', fontweight='bold')
ax.set_xlabel('Camada')
ax.set_ylabel('Acuracia (%)')
ax.set_xticks(x)
ax.set_xticklabels(vmamba_camadas, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pasta + '/curva_vmamba.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: curva_vmamba.png')

print('\nPronto!')
