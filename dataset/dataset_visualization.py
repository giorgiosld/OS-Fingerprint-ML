import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_improved_visualizations(file_path):
    df = pd.read_csv(file_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Log-scale violin plot
    sns.violinplot(x='label', y='pointer_graph_length', data=df, ax=ax1)
    ax1.set_yscale('log')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_title('Violin Plot (Log Scale)')
    
    # 2. Filtered strip plot (showing only points below 95th percentile)
    threshold = np.percentile(df['pointer_graph_length'], 95)
    df_filtered = df[df['pointer_graph_length'] <= threshold]
    sns.stripplot(x='label', y='pointer_graph_length', data=df_filtered, 
                 jitter=True, alpha=0.4, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_title(f'Strip Plot (Below 95th percentile: {threshold:.0f})')
    
    # 3. KDE plot with log scale
    for version in df['label'].unique():
        subset = df[df['label'] == version]['pointer_graph_length']
        sns.kdeplot(data=np.log1p(subset), label=version, ax=ax3)
    ax3.set_title('KDE Plot (Log-transformed)')
    ax3.set_xlabel('log(pointer_graph_length + 1)')
    ax3.legend()
    
    # 4. Box plot with log scale
    sns.boxplot(x='label', y='pointer_graph_length', data=df, ax=ax4)
    ax4.set_yscale('log')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.set_title('Box Plot (Log Scale)')
    
    plt.tight_layout()
    plt.savefig('statistical_visualizations.png')
    plt.close()

# Run visualization
create_improved_visualizations('dataset_200_full.csv')
