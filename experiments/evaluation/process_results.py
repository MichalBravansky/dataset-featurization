import pandas as pd
import numpy as np
import os
import json
import datasets
import random
from typing import Dict, List
from tqdm import tqdm
from experiments.utils.semantic_similarity import SemanticSimilarityEvaluator
from experiments.utils.model_reconstruction import ModelReconstructionEvaluator
from experiments.utils.correlation_coverage import CorrelationCoverageEvaluator

def load_features(dataset_name: str, group: int) -> Dict[str, List[str]]:
    """Load features for each method from features.csv"""
    features_df = pd.read_csv("experiments/data/features.csv")
    features_df = features_df[
        (features_df['group'] == group) & 
        (features_df['dataset'] == dataset_name)
    ]

    methods = {
        'baseline': features_df[features_df["type"] == "baseline"].sort_values(by="id")["feature"].tolist(),
        'prompting': features_df[features_df["type"] == "prompting"].sort_values(by="id")["feature"].tolist(),
        'clustering': features_df[features_df["type"] == "clustering"].sort_values(by="id")["feature"].tolist(),
        'featurization': features_df[features_df["type"] == "featurization"].sort_values(by="id")["feature"].tolist()
    }
    return methods

def process_group_data(dataset_name: str, group: int) -> pd.DataFrame:
    """Process data for a single group of a dataset"""

    semantic_evaluator = SemanticSimilarityEvaluator()
    model_evaluator = ModelReconstructionEvaluator()
    correlation_evaluator = CorrelationCoverageEvaluator()
    
    verified_df = pd.read_csv(f"experiments/data/{dataset_name}/evaluation_df_{group}_all.csv")

    methods = load_features(dataset_name, group)
    
    sample_df = datasets.load_dataset(
        "Bravansky/dataset-featurization",
        dataset_name,
        split="train"
    ).to_pandas()

    sample_df = sample_df[sample_df['group'] == group]
    sample_df["category"] = sample_df["category"].apply(lambda x: x.split('/')[-1])
    
    verified_df = verified_df.merge(sample_df, left_on = ['string'], right_on =['text'], how='inner')
    verified_df.drop(columns=['text'], inplace=True)

    categories = list(verified_df["category"].unique())
    
    results = []
    
    for method_name, features in tqdm(methods.items(), desc=f"Processing {dataset_name} group {group}"):
        valid_features = [f for f in features if f in verified_df.columns]
        if not valid_features:
            continue

        similarities = semantic_evaluator.get_similarities(verified_df, valid_features, categories)

        for top_n in range(1, 51):
            current_features = valid_features[:top_n]

            model_metrics = model_evaluator.evaluate(verified_df, current_features, top_n)

            correlation = correlation_evaluator.evaluate(verified_df, current_features, categories, top_n)

            current_similarities = np.array(similarities[:top_n])
            max_similarities = np.max(current_similarities, axis=0)
            avg_similarity = np.mean(max_similarities) * 5
            
            results.append({
                'dataset': dataset_name,
                'method': method_name,
                'group': group,
                'top_features': top_n,
                'class_coverage': correlation,
                'reconstruction_accuracy': model_metrics['mean_cv_score'],
                'reconstruction_std': model_metrics['std_cv_score'],
                'semantic_preservation': avg_similarity
            })
    
    return pd.DataFrame(results)

def main():
    datasets = ['nyt', 'amazon', 'dbpedia']
    groups = [0, 1, 2]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        for group in groups:
            print(f"  Group {group}")
            results = process_group_data(dataset, group)
            all_results.append(results)

    combined_df = pd.concat(all_results, ignore_index=True)

    combined_df.to_csv('experiments/data/metrics_calculated.csv', index=False)

if __name__ == "__main__":
    main() 