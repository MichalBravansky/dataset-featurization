import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def calculate_rewards_batch(args):
    """Vectorized reward calculation for a batch of bootstrapped data"""
    normalized_scores, test_col_idx, train_col_idx, prompt_groups, batch_size = args
    gaps = []
    
    for group in prompt_groups:
        batch_indices = np.random.choice(group, size=min(batch_size, len(group)), replace=True)
        batch_test_scores = normalized_scores[batch_indices, test_col_idx]
        best_idx = batch_indices[np.argmax(batch_test_scores)]
        gaps.append(normalized_scores[best_idx, train_col_idx])
    
    return np.mean(gaps)

colors = sns.color_palette("colorblind")

legend_labels = {
    'cpm_50': 'Baseline CPM (14 features)',
    'hh_5': 'Our Method (5 features)',
    'hh_14': 'Our Method (14 features)',
    'hh_50': 'Our Method (50 features)',
    'shp_5': 'Our Method (5 features)',
    'shp_14': 'Our Method (14 features)',
    'shp_50': 'Our Method (50 features)',
    'reference_pm': 'Reference Model'
}

colors = {
    'cpm_50': colors[0],
    'hh_5': colors[1],
    'hh_14': colors[2],
    'hh_50': colors[3],
    'shp_5': colors[1],
    'shp_14': colors[2],
    'shp_50': colors[3],
    'reference_pm': colors[7]
}

def main():
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'wspace': 0.2})

    for idx, dataset_type in enumerate(['shp', 'hh']): ##hh

        results_df = pd.read_feather(f'out/{dataset_type}/results/reward_generated_results.feather')
        results_df.dropna(inplace=True)
        results_df = results_df.groupby("prompt").head(16)
        results_df.reset_index(drop=True, inplace=True)
        
        if dataset_type == 'hh':
            results_df = pd.concat([results_df, pd.read_parquet("data/hh-generated-results.parquet")["reference_pm"]], axis = 1)
        else:
            reference_df = pd.read_parquet("data/shp-external-reward-2.parquet")[["prompt", "response", "reference_pm"]]
            reference_df = reference_df.drop_duplicates(subset=["prompt", "response"]) 
            results_df = results_df.merge(reference_df[["reference_pm", "prompt", "response"]], on=["prompt", "response"], how="left")
            print(results_df.isna().sum())

        test_columns = {"_".join(col.rsplit("_")[-2:]): col for col in results_df.columns if 'test' in col.lower()}
        test_columns['reference_pm'] = 'reference_pm'
        train_columns = {"_".join(col.rsplit("_")[-2:]): col for col in results_df.columns if 'train' in col.lower()}
        train_columns['reference_pm'] = 'reference_pm'
        
        results_df = results_df.groupby("prompt").head(16)
        results_df.reset_index(drop=True, inplace=True)

        cols_to_normalize = list(test_columns.values()) + list(train_columns.values())
        cols_to_normalize = list(set(cols_to_normalize))  # Remove duplicates
        normalized_scores = np.zeros((len(results_df), len(cols_to_normalize)))

        for i, col in enumerate(cols_to_normalize):
            values = results_df[col].values
            normalized_scores[:, i] = (values - values.mean()) / values.std()

        prompt_groups = [group.index.values for _, group in results_df.groupby('prompt')]
        
        batch_sizes = range(1, 17)
        n_bootstrap = 500
        results = {test_name: [] for test_name in test_columns.keys()}

        for test_name, test_col in test_columns.items():
            rewards = []
            errors = []
            
            test_col_idx = cols_to_normalize.index(test_col)
            train_col_idx = cols_to_normalize.index(train_columns["reference_pm"])
            
            for batch_size in tqdm(batch_sizes, desc=f"{dataset_type}-{test_name}"):
                args_list = [(normalized_scores, test_col_idx, train_col_idx, prompt_groups, batch_size) 
                            for _ in range(n_bootstrap)]

                n_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                    bootstrap_rewards = list(executor.map(calculate_rewards_batch, args_list))
                
                rewards.append(np.mean(bootstrap_rewards))
                errors.append(np.std(bootstrap_rewards))
            
            results[test_name] = {'mean': rewards, 'std': errors}

        # Save results to parquet file
        results_df = pd.DataFrame()
        for test_name, reward_data in results.items():
            results_df[f'{test_name}_mean'] = reward_data['mean']
            results_df[f'{test_name}_std'] = reward_data['std']
        results_df['batch_sizes'] = batch_sizes
        results_df.to_parquet(f'robustness_results_{dataset_type}.parquet')

        for test_name, reward_data in results.items():
            label = legend_labels.get(test_name, test_name)
            color = colors[test_name]
            
            mean = reward_data['mean']
            std = reward_data['std']

            axes[idx].plot(batch_sizes, mean, marker='o', linewidth=2, markersize=8,
                          label=label, color=color)

            axes[idx].fill_between(batch_sizes, 
                                 np.array(mean) - np.array(std),
                                 np.array(mean) + np.array(std),
                                 color=color, alpha=0.2)

        title = 'HH-RLHF' if dataset_type == 'hh' else dataset_type.upper()
        axes[idx].set_title(title, fontsize=14)

        axes[idx].grid(True, which='major', linewidth=0.5, color='lightgray')
        axes[idx].grid(True, which='minor', linewidth=0.3, color='lightgray', alpha=0.5)
        axes[idx].set_axisbelow(True)
        axes[idx].tick_params(axis='both', labelsize=12)

    fig.text(0.5, 0.02, 'Number of responses in BoN', fontsize=14, ha='center')
    fig.text(0.08, 0.5, 'PM Score', fontsize=14, va='center', rotation='vertical')

    unique_legend_items = {
        'cpm_50': 'Baseline CPM (14 features)',
        'reference_pm': 'Reference PM',
        'hh_5': 'Our CPM (5 features)',
        'hh_14': 'Our CPM (14 features)',
        'hh_50': 'Our CPM (50 features)'
    }

    legend_lines = [plt.Line2D([0], [0], color=colors[key], marker='o', linestyle='-', 
                              linewidth=2, markersize=8) 
                   for key in unique_legend_items.keys()]
    legend_labels_list = list(unique_legend_items.values())

    fig.legend(legend_lines, 
              legend_labels_list,
              fontsize=14,
              bbox_to_anchor=(0.5, 0.97),
              loc='center',
              ncol=5,
              borderaxespad=0,
              frameon=False,
              edgecolor='black',
              bbox_transform=fig.transFigure)

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.90])

    plt.savefig('robustness_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()