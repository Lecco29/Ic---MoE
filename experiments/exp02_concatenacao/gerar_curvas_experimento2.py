
#Gera visualizações do Experimento 2: Fusão Early-Deep.

import os
import numpy as np
import matplotlib.pyplot as plt

# aumenta as letras (pra ficar legivel no artigo)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 22,
})

# Dados do Experimento 2 

exp2_data = {
    'vgg16': {
        'color': {'E': 95.13, 'E_std': 0.81, 'D': 71.80, 'D_std': 2.55, 'Z': 86.13, 'Z_std': 1.86},
        'texture': {'E': 70.73, 'E_std': 2.25, 'D': 94.73, 'D_std': 0.74, 'Z': 94.73, 'Z_std': 0.90}
    },
    'resnet50': {
        'color': {'E': 95.87, 'E_std': 0.50, 'D': 88.27, 'D_std': 1.12, 'Z': 94.13, 'Z_std': 0.78},
        'texture': {'E': 90.67, 'E_std': 1.65, 'D': 93.27, 'D_std': 0.25, 'Z': 93.53, 'Z_std': 1.29}
    },
    'ibot': {
        'color': {'E': 96.93, 'E_std': 0.49, 'D': 81.33, 'D_std': 1.62, 'Z': 85.13, 'Z_std': 1.69},
        'texture': {'E': 78.33, 'E_std': 0.92, 'D': 87.80, 'D_std': 1.36, 'Z': 88.07, 'Z_std': 1.04}
    },
    'vmamba': {
        'color': {'E': 94.80, 'E_std': 0.62, 'D': 93.93, 'D_std': 0.25, 'Z': 94.13, 'Z_std': 0.16},
        'texture': {'E': 88.40, 'E_std': 1.51, 'D': 94.33, 'D_std': 1.40, 'Z': 93.40, 'Z_std': 1.96}
    }
}

backbone_names = {
    'vgg16': 'VGG-16',
    'resnet50': 'ResNet-50',
    'ibot': 'iBOT',
    'vmamba': 'VMamba'
}


def plot_fusion_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    backbones = ['vgg16', 'resnet50', 'ibot', 'vmamba']

    for idx, backbone in enumerate(backbones):
        ax = axes[idx // 2, idx % 2]
        data = exp2_data[backbone]

        x = np.arange(2)  # color, texture
        width = 0.25

        # E (cor layer)
        e_vals = [data['color']['E'], data['texture']['E']]
        e_stds = [data['color']['E_std'], data['texture']['E_std']]

        # D (texture layer)
        d_vals = [data['color']['D'], data['texture']['D']]
        d_stds = [data['color']['D_std'], data['texture']['D_std']]

        # Z (fusão)
        z_vals = [data['color']['Z'], data['texture']['Z']]
        z_stds = [data['color']['Z_std'], data['texture']['Z_std']]

        bars1 = ax.bar(x - width, e_vals, width, yerr=e_stds, label='E (color layer)',
                       color='#e74c3c', capsize=3, alpha=0.8)
        bars2 = ax.bar(x, d_vals, width, yerr=d_stds, label='D (texture layer)',
                       color='#3498db', capsize=3, alpha=0.8)
        bars3 = ax.bar(x + width, z_vals, width, yerr=z_stds, label='Z (fusion)',
                       color='#2ecc71', capsize=3, alpha=0.8)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(backbone_names[backbone], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Color Task', 'Texture Task'])
        ax.set_ylim(60, 100)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

        # Adiciona valores nas barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=12)

    plt.suptitle('Experiment 2: Early-Deep Fusion (E vs D vs Z)', fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_improvement():
    fig, ax = plt.subplots(figsize=(12, 6))

    backbones = ['vgg16', 'resnet50', 'ibot', 'vmamba']

    x = np.arange(len(backbones))
    width = 0.35

    # Calcula melhora para cor e textura
    color_improvement = []
    texture_improvement = []

    for backbone in backbones:
        data = exp2_data[backbone]
        # Cor: compara Z com melhor entre E e D
        best_color = max(data['color']['E'], data['color']['D'])
        color_improvement.append(data['color']['Z'] - best_color)

        # Textura: compara Z com melhor entre E e D
        best_texture = max(data['texture']['E'], data['texture']['D'])
        texture_improvement.append(data['texture']['Z'] - best_texture)

    bars1 = ax.bar(x - width/2, color_improvement, width, label='Color Task', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, texture_improvement, width, label='Texture Task', color='#3498db', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Improvement over best single layer (%)')
    ax.set_xlabel('Backbone')
    ax.set_title('Fusion Impact: Does Z improve over E or D?', fontsize=26, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([backbone_names[b] for b in backbones])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-15, 5)

    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -10
            ax.annotate(f'{height:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def plot_summary_table():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Dados da tabela
    headers = ['Backbone', 'Task', 'E (color)', 'D (texture)', 'Z (fusion)', 'Best', 'Δ']

    rows = []
    for backbone in ['vgg16', 'resnet50', 'ibot', 'vmamba']:
        data = exp2_data[backbone]
        name = backbone_names[backbone]

        # Color task
        e_c = data['color']['E']
        d_c = data['color']['D']
        z_c = data['color']['Z']
        best_c = 'E' if e_c >= d_c else 'D'
        delta_c = z_c - max(e_c, d_c)
        rows.append([name, 'Color', f'{e_c:.1f}%', f'{d_c:.1f}%', f'{z_c:.1f}%', best_c, f'{delta_c:+.1f}%'])

        # Texture task
        e_t = data['texture']['E']
        d_t = data['texture']['D']
        z_t = data['texture']['Z']
        best_t = 'E' if e_t >= d_t else 'D'
        delta_t = z_t - max(e_t, d_t)
        rows.append(['', 'Texture', f'{e_t:.1f}%', f'{d_t:.1f}%', f'{z_t:.1f}%', best_t, f'{delta_t:+.1f}%'])

    # Cria tabela
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.10, 0.12, 0.12, 0.12, 0.08, 0.10]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Estiliza header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Colore células baseado no valor de Δ
    for row_idx in range(1, len(rows) + 1):
        delta_val = float(rows[row_idx-1][6].replace('%', '').replace('+', ''))
        if delta_val < 0:
            table[(row_idx, 6)].set_facecolor('#ffcccc')
        elif delta_val > 0:
            table[(row_idx, 6)].set_facecolor('#ccffcc')
        else:
            table[(row_idx, 6)].set_facecolor('#ffffcc')

    ax.set_title('Experiment 2: Fusion Results Summary\n(Δ = Z - best(E,D))',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    return fig


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Gerando visualizações do Experimento 2 (Fusão)...")

    # Figura 1, Comparação E vs D vs Z
    fig1 = plot_fusion_comparison()
    path1 = os.path.join(output_dir, 'exp2_fusao_comparacao.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Salvo: {path1}")
    plt.close(fig1)

    # Figura 2, Melhora/piora da fusão
    fig2 = plot_improvement()
    path2 = os.path.join(output_dir, 'exp2_fusao_improvement.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Salvo: {path2}")
    plt.close(fig2)

    # Figura 3, Tabela resumo
    fig3 = plot_summary_table()
    path3 = os.path.join(output_dir, 'exp2_fusao_tabela.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Salvo: {path3}")
    plt.close(fig3)



if __name__ == '__main__':
    main()
