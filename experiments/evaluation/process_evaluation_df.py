import pandas as pd
import random
import sys
sys.path.append("dataset_featurization")
from utils.verifier import Verifier
import os
from datasets import load_dataset

verifier = Verifier()
random.seed(42)

def load_and_verify_features(folder_name, group_number):
    """Load features from features.csv and verify them against HuggingFace evaluation data"""
    
    features_file = "experiments/data/features.csv"
    features_df = pd.read_csv(features_file)

    features_df = features_df[
        (features_df['dataset'] == folder_name) & 
        (features_df['group'] == group_number)
    ]

    evaluation_df = load_dataset(
        "Bravansky/dataset-featurization", 
        f"{folder_name}-evaluation-{group_number}", 
        split="train"
    ).to_pandas()

    features_to_evaluate = features_df[(features_df['type'] == "baseline") | (features_df['type'] == "prompting")]["feature"].tolist()
    features_to_evaluate = [feature for feature in features_to_evaluate if feature not in evaluation_df.columns]

    verification_df = verifier.process(evaluation_df['string'], features_to_evaluate)

    merged_df = pd.merge(evaluation_df, verification_df, on='string', how='inner')
    
    os.makedirs(f"experiments/data/{folder_name}", exist_ok=True)

    output_file = f"experiments/data/{folder_name}/evaluation_df_{group_number}_all.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    merged_df.to_csv(output_file, index=False)

def main():
    datasets = ['amazon', 'nyt', 'dbpedia']
    groups = [0, 1, 2]
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        for group in groups:
            load_and_verify_features(dataset, group)

if __name__ == "__main__":
    main()