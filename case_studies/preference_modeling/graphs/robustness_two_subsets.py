import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

configs = {
    'hh': {
        'cpm_50': 'Baseline (14 features)',
        'hh_50': 'Our Method (50 features)',
        'hh_14': 'Our Method (14 features)',
        'hh_5': 'Our Method (5 features)',
    },
    'shp': {
        'cpm_50': 'Baseline (14 features)',
        'shp_50': 'Our Method (50 features)',
        'shp_14': 'Our Method (14 features)',
        'shp_5': 'Our Method (5 features)',
    }
}

def normalize_scores(df, col, train_mean=None, train_std=None):
    """Normalize scores using training statistics if provided"""
    if train_mean is None:
        train_mean = df[col].mean()
    if train_std is None:
        train_std = df[col].std()
    return (df[col] - train_mean) / train_std

def calculate_rewards(df, test_col, train_col, batch_size):
    df = df.copy()

    test_mean = df[test_col].mean()
    test_std = df[test_col].std()
    df[test_col] = normalize_scores(df, test_col, test_mean, test_std)
    
    train_mean = df[train_col].mean()
    train_std = df[train_col].std()
    df[train_col] = normalize_scores(df, train_col, train_mean, train_std)
    
    test_scores = []
    train_scores = []

    for _, group in df.groupby('prompt'):
        batch = group.head(batch_size)

        best_response_idx = batch[test_col].idxmax()

        test_score = batch.loc[best_response_idx, test_col]
        train_score = batch.loc[best_response_idx, train_col]
        
        test_scores.append(test_score)
        train_scores.append(train_score)
    
    return np.mean(test_scores), np.mean(train_scores)

def calculate_rewards_batch(args):
    """Vectorized reward calculation for a batch of bootstrapped data"""
    normalized_scores, test_col_idx, train_col_idx, prompt_groups, batch_size = args
    test_scores = []
    train_scores = []
    
    for group in prompt_groups:
        batch_indices = np.random.choice(group, size=min(batch_size, len(group)), replace=True)
        batch_test_scores = normalized_scores[batch_indices, test_col_idx]
        best_idx = batch_indices[np.argmax(batch_test_scores)]
        
        test_scores.append(normalized_scores[best_idx, test_col_idx])
        train_scores.append(normalized_scores[best_idx, train_col_idx])
    
    return np.mean(test_scores), np.mean(train_scores)

colors = sns.color_palette("colorblind")

legend_labels = {
    'cpm_50': 'Baseline (14 features)',
    'hh_5': 'Our Method (5 features)',
    'hh_14': 'Our Method (14 features)',
    'hh_50': 'Our Method (50 features)',
    'shp_5': 'Our Method (5 features)',
    'shp_14': 'Our Method (14 features)',
    'shp_50': 'Our Method (50 features)',
}

colors = {
    'cpm_50': colors[0],
    'hh_5': colors[1],
    'hh_14': colors[2],
    'hh_50': colors[3],
    'shp_5': colors[1],
    'shp_14': colors[2],
    'shp_50': colors[3],
}

fig, axes = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'wspace': 0.2})

batch_sizes = range(1, 17)

all_scores = []
for dataset in ['hh', 'shp']:
    results_df = pd.read_feather(f'out/{dataset}/results/reward_generated_results.feather')
    results_df.dropna(inplace=True)
    
    for config in configs[dataset].keys():
        if dataset == 'hh':
            test_col = f'score_hh-rlhf-test-results_{config}'
            train_col = f'score_hh-rlhf-results_train_{config}'
        else:
            test_col = f'score_shp-with-features-test-results_{config}'
            train_col = f'score_shp-with-features-results_train_{config}'
            
        for batch_size in batch_sizes:
            test_score, train_score = calculate_rewards(results_df, test_col, train_col, batch_size)
            all_scores.extend([test_score, train_score])

y_min = min(all_scores) - 0.1
y_max = max(all_scores) + 0.1

def main():
    for dataset_idx, dataset in enumerate(['hh', 'shp']):
        dataset_configs = configs[dataset]
        data_dir = dataset
        results_df = pd.read_feather(f'out/{data_dir}/results/reward_generated_results.feather')
        results_df.dropna(inplace=True)

        results_df = results_df.groupby("prompt").head(16)
        results_df.reset_index(drop=True, inplace=True)
        
        cols_to_normalize = []
        for config in dataset_configs.keys():
            if dataset == 'hh':
                cols_to_normalize.extend([
                    f'score_hh-rlhf-test-results_{config}',
                    f'score_hh-rlhf-results_train_{config}'
                ])
            else:
                cols_to_normalize.extend([
                    f'score_shp-with-features-test-results_{config}',
                    f'score_shp-with-features-results_train_{config}'
                ])
        
        normalized_scores = np.zeros((len(results_df), len(cols_to_normalize)))
        for i, col in enumerate(cols_to_normalize):
            values = results_df[col].values
            normalized_scores[:, i] = (values - values.mean()) / values.std()

        prompt_groups = [group.index.values for _, group in results_df.groupby('prompt')]
        
        ax = axes[dataset_idx]
        results_dict = {}
        
        for config, label in tqdm(dataset_configs.items(), desc=f"{dataset} configs"):
            if dataset == 'hh':
                test_col = f'score_hh-rlhf-test-results_{config}'
                train_col = f'score_hh-rlhf-results_train_{config}'
            else:
                test_col = f'score_shp-with-features-test-results_{config}'
                train_col = f'score_shp-with-features-results_train_{config}'
            
            test_col_idx = cols_to_normalize.index(test_col)
            train_col_idx = cols_to_normalize.index(train_col)
            
            test_means, test_stds = [], []
            train_means, train_stds = [], []
            
            for batch_size in tqdm(batch_sizes, desc=f"Batch sizes for {config}", leave=False):
                args_list = [(normalized_scores, test_col_idx, train_col_idx, prompt_groups, batch_size) 
                            for _ in range(500)]
                
                n_workers = max(1, multiprocessing.cpu_count() - 1)
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                    bootstrap_results = list(executor.map(calculate_rewards_batch, args_list))
                
                test_scores, train_scores = zip(*bootstrap_results)
                
                test_means.append(np.mean(test_scores))
                test_stds.append(np.std(test_scores))
                train_means.append(np.mean(train_scores))
                train_stds.append(np.std(train_scores))
            
            results_dict[config] = {
                'test_mean': test_means,
                'test_std': test_stds,
                'train_mean': train_means,
                'train_std': train_stds
            }

            ax.plot(batch_sizes, test_means, marker='o', color=colors[config], 
                   linestyle='-', linewidth=2, markersize=8, alpha=0.8, zorder=2)
            ax.fill_between(batch_sizes, 
                          np.array(test_means) - np.array(test_stds),
                          np.array(test_means) + np.array(test_stds),
                          color=colors[config], alpha=0.2)
            
            ax.plot(batch_sizes, train_means, marker='o', color=colors[config],
                   linestyle='--', linewidth=2, markersize=8, alpha=0.8, zorder=2)
            ax.fill_between(batch_sizes,
                          np.array(train_means) - np.array(train_stds),
                          np.array(train_means) + np.array(train_stds),
                          color=colors[config], alpha=0.2)

        results_df = pd.DataFrame()
        for config in dataset_configs:
            results_df[f'{config}_test_mean'] = results_dict[config]['test_mean']
            results_df[f'{config}_test_std'] = results_dict[config]['test_std']
            results_df[f'{config}_train_mean'] = results_dict[config]['train_mean']
            results_df[f'{config}_train_std'] = results_dict[config]['train_std']
        results_df['batch_sizes'] = batch_sizes
        results_df.to_parquet(f'robustness_first_results_{dataset}.parquet')

    for dataset_idx, dataset in enumerate(['hh', 'shp']):
        title = 'HH-RLHF' if dataset == 'hh' else dataset.upper()
        axes[dataset_idx].set_title(title, fontsize=14)
        axes[dataset_idx].grid(True, which='major', linewidth=0.5, color='lightgray', alpha=0.5, zorder=1)
        axes[dataset_idx].grid(True, which='minor', linewidth=0.3, color='lightgray', alpha=0.3, zorder=1)
        axes[dataset_idx].set_axisbelow(True)
        axes[dataset_idx].tick_params(axis='both', labelsize=12)
        axes[dataset_idx].set_ylim(y_min, y_max)

        if dataset_idx == 0:
            axes[dataset_idx].set_ylabel('PM Score', fontsize=14)

    fig.text(0.5, 0.02, 'Number of responses in BoN', fontsize=14, ha='center')

    unique_legend_items = {
        'cpm_50': 'Baseline CPM (14 features)',
        'hh_5': 'Our CPM (5 features)',
        'hh_14': 'Our CPM (14 features)',
        'hh_50': 'Our CPM (50 features)'
    }

    method_lines = [plt.Line2D([0], [0], color=colors[key],
                              linestyle='-', linewidth=3, markersize=8, alpha=0.5)
                    for key in unique_legend_items.keys()]
    method_labels = list(unique_legend_items.values())

    pm_lines = [
        plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=3, alpha=0.5),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, alpha=0.5)
    ]
    pm_labels = ['PM A (used for argmax)', 'PM B']

    fig.legend(method_lines + pm_lines,
              method_labels + pm_labels,
              fontsize=13,
              bbox_to_anchor=(0.5, 0.963),
              loc='center',
              ncol=3,
              borderaxespad=0,
              frameon=False,
              edgecolor='black',
              bbox_transform=fig.transFigure)

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.90])

    plt.savefig('robustness_comparison_graph.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()