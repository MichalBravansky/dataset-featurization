import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

colors = sns.color_palette("colorblind")

colors_dict = {
    'cpm_50': colors[0],
    'hh_5': colors[1],
    'hh_14': colors[2],
    'hh_50': colors[3],
    'shp_5': colors[1],
    'shp_14': colors[2],
    'shp_50': colors[3],
    'reference_pm': colors[7]
}

fig, axes = plt.subplots(1, 4, figsize=(14, 4), gridspec_kw={'wspace': 0.15, 'left': 0.06, 'right': 0.99})

legend_labels = {
    'cpm_50': 'Baseline CPM (14 features)',
    'reference_pm': 'Reference PM',
    'hh_5': 'Our CPM (5 features)',
    'hh_14': 'Our CPM (14 features)',
    'hh_50': 'Our CPM (50 features)',
    'shp_5': 'Our CPM (5 features)',
    'shp_14': 'Our CPM (14 features)',
    'shp_50': 'Our CPM (50 features)'
}

for dataset_idx, dataset in enumerate(['hh', 'shp']):
    results_df = pd.read_parquet(f'robustness_results_{dataset}.parquet')
    ax = axes[dataset_idx]
    
    for col in results_df.columns:
        if col == 'batch_sizes':
            continue
        if col.endswith('_std'):
            continue
            
        test_name = col.replace('_mean', '')
        if test_name not in legend_labels:
            continue
            
        mean = results_df[col]
        std = results_df[f'{test_name}_std']
        
        ax.plot(results_df['batch_sizes'], mean, 
                marker='o', 
                linewidth=2, 
                markersize=8,
                color=colors_dict[test_name])
        
        ax.fill_between(results_df['batch_sizes'],
                       mean - std,
                       mean + std,
                       color=colors_dict[test_name],
                       alpha=0.2)
    
    title = 'HH-RLHF' if dataset == 'hh' else dataset.upper()
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='major', linewidth=0.5, color='lightgray')
    ax.grid(True, which='minor', linewidth=0.3, color='lightgray', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('')

fig.text(0.25, 0.01, 'Number of responses in BoN', fontsize=12, ha='center')

for dataset_idx, dataset in enumerate(['hh', 'shp']):
    results_df = pd.read_parquet(f'robustness_first_results_{dataset}.parquet')
    
    ax = axes[dataset_idx + 2]
    bar_width = 0.8
    x_positions = np.arange(4)
    
    configs = ['cpm_50', f'{dataset}_5', f'{dataset}_14', f'{dataset}_50']
    
    for i, config in enumerate(configs):
        train_mean = results_df[f'{config}_train_mean'].iloc[0]
        train_std = results_df[f'{config}_train_std'].iloc[0]
        test_mean = results_df[f'{config}_test_mean'].iloc[0]
        test_std = results_df[f'{config}_test_std'].iloc[0]
        
        ax.bar(x_positions[i], train_mean, bar_width,
               color=colors_dict[config], alpha=1.0)
        
        difference = test_mean - train_mean
        if difference > 0:
            ax.bar(x_positions[i], difference, bar_width,
                  bottom=train_mean, color='gray', alpha=0.3,
                  hatch='///', edgecolor='black', linewidth=1)
        
        ax.errorbar(x_positions[i], train_mean, yerr=train_std,
                   color='black', capsize=5, fmt='none', alpha=0.5)
        ax.errorbar(x_positions[i], test_mean, yerr=test_std,
                   color='black', capsize=5, fmt='none', alpha=0.5)
    
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks([])
    ax.set_xticklabels([])
    title = 'HH-RLHF' if dataset == 'hh' else dataset.upper()
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='major', linewidth=0.5, color='lightgray', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=12)

    solid_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=1.0)
    hatch_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3, hatch='///', edgecolor='black')
    
    ax.legend([hatch_patch, solid_patch], 
              ['PM A', 'PM B'],
              loc='lower right',
              bbox_to_anchor=(0.98, 0.02),
              frameon=True,
              fontsize=10,
              title='Models')

fig.text(0.75, 0.01, 'Different Feature Sets For BoN = 16', fontsize=12, ha='center')

method_lines = [plt.Line2D([0], [0], color=colors_dict[key],
                          linestyle='-', linewidth=3, markersize=8, alpha=1.0)
                for key in ['cpm_50', 'hh_5', 'hh_14', 'hh_50', 'reference_pm']]
method_labels = [legend_labels[key] for key in ['cpm_50', 'hh_5', 'hh_14', 'hh_50', 'reference_pm']]

fig.legend(method_lines,
          method_labels,
          fontsize=11,
          bbox_to_anchor=(0.5, 0.98),
          loc='center',
          ncol=5,
          columnspacing=0.8,
          handletextpad=0.4,
          borderaxespad=0,
          frameon=False,
          bbox_transform=fig.transFigure)

fig.text(0.01, 0.5, 'PM Score', fontsize=14, va='center', rotation='vertical')

plt.tight_layout(rect=[0.06, 0.05, 0.99, 0.92])

plt.savefig('combined_robustness_comparison.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
plt.show()