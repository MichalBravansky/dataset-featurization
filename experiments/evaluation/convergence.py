import pandas as pd
import numpy as np

def analyze_convergence(df, target_method='Full Pipeline', baseline_method='Baseline', threshold=0.95):
    """
    Analyze how quickly methods converge to their maximum performance.
    For each group, calculate convergence point and then average across groups.
    
    Args:
        df: DataFrame with columns 'method', 'top_features', 'group', and 'score'
        target_method: The method to compare against
        baseline_method: The baseline method
        threshold: The fraction of max performance to consider as "converged"
    """
    results = {}
    for method in [target_method, baseline_method]:
        convergence_points = []
        method_groups = df[df['method'] == method]['group'].unique()
        
        for group in method_groups:
            group_data = df[(df['method'] == method) & (df['group'] == group)]
            max_score = group_data['score'].max()
            threshold_score = max_score * threshold
            
            convergence_point = None
            for idx, row in group_data.iterrows():
                if row['score'] >= threshold_score:
                    remaining_scores = group_data[group_data['top_features'] >= row['top_features']]['score']
                    if all(remaining_scores >= threshold_score):
                        convergence_point = row
                        break
            
            if convergence_point is not None:
                convergence_points.append(int(convergence_point['top_features']))
            else:
                print(f"Warning: Group {group} for method {method} never converges to {threshold*100}% of max performance")
        
        if not convergence_points:
            raise ValueError(f"Method {method} never converges to {threshold*100}% of max performance in any group")
            
        results[method] = {
            'convergence_k': np.mean(convergence_points),
            'convergence_k_std': np.std(convergence_points) if len(convergence_points) > 1 else 0,
            'n_groups': len(convergence_points)
        }

    speed_ratio = results[baseline_method]['convergence_k'] / results[target_method]['convergence_k']
    results['speed_ratio'] = speed_ratio
    return results

def main():
    df = pd.read_csv("experiments/data/metrics_calculated.csv")

    method_map = {
        'baseline': 'Baseline',
        'prompting': 'Generation Stage',
        'clustering': 'Clustering Stage',
        'featurization': 'Full Pipeline'
    }

    df['method'] = df['method'].map(method_map)
    
    datasets = ['nyt', 'amazon', 'dbpedia']
    metrics = {
        'Category Coverage': 'class_coverage',
        'Reconstruction Accuracy': 'reconstruction_accuracy',
        'Semantic Preservation': 'semantic_preservation'
    }
    
    print("\nConvergence Analysis (95% threshold)")
    print("====================================")
    
    all_ratios = []
    
    for metric_name, column_name in metrics.items():
        print(f"\n{metric_name}:")
        print("-" * len(metric_name) + "-")
        
        metric_ratios = []
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            
            # Prepare data for convergence analysis
            metric_df = dataset_df.copy()
            metric_df['score'] = metric_df[column_name]
            
            try:
                results = analyze_convergence(metric_df)
                
                print(f"\n{dataset.upper()}:")
                print(f"Full Pipeline converges at k={results['Full Pipeline']['convergence_k']:.1f} ± {results['Full Pipeline']['convergence_k_std']:.1f} features (n={results['Full Pipeline']['n_groups']})")
                print(f"Baseline converges at k={results['Baseline']['convergence_k']:.1f} ± {results['Baseline']['convergence_k_std']:.1f} features (n={results['Baseline']['n_groups']})")
                print(f"Ratio: Baseline needs {results['speed_ratio']:.1f}x more features")
                
                metric_ratios.append(results['speed_ratio'])
                all_ratios.append(results['speed_ratio'])
            except ValueError as e:
                print(f"\n{dataset.upper()}: {str(e)}")
                continue
        
        if metric_ratios:
            avg_ratio = np.mean(metric_ratios)
            print(f"\nAverage ratio for {metric_name}: {avg_ratio:.1f}x")
    
    if all_ratios:
        overall_avg = np.mean(all_ratios)
        print("\nOverall Statistics")
        print("=================")
        print(f"Average across all metrics and datasets: {overall_avg:.1f}x")
        print(f"Range: {min(all_ratios):.1f}x - {max(all_ratios):.1f}x")

if __name__ == "__main__":
    main() 