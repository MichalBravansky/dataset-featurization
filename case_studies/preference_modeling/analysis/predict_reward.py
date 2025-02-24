import pandas as pd
import dill as pickle
from datasets import load_dataset, load_from_disk
from pathlib import Path
import sys
sys.path.append('..')
from utils import features_values, add_model_scores

def standardize(data):
    """Standardize the data to have zero mean and unit variance"""
    data -= data.mean()
    data /= data.std()
    return data

def load_model_scores(data_df, model_path):
    """Load and apply model scores based on file type"""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_name = Path(model_path).stem

        features_matrix = features_values(data_df, model['feature_names'])
        scores = add_model_scores(model_name, data_df, model, features_matrix, standardize=True)
        return scores[model_name], model_name
        
    elif model_path.endswith('.json'):
        scores_df = pd.read_json(model_path)
        scores_df = scores_df.drop_duplicates()
        scores = standardize(scores_df['score'].values)
        return scores, 'pm_score'
    
    else:
        try:
            scores_df = load_dataset(model_path)['train'].to_pandas()
        except:
            scores_df = load_from_disk(model_path)['train'].to_pandas()
        scores_df = scores_df.drop_duplicates()
        scores = standardize(scores_df['external_rm1'].values)
        return scores, 'external_rm1'

def main():

    datasets = {
        "shp": {
            "data": "shp-bon",
            "models": [
                "models/shp/shp-test_cpm_50.pkl",
                "models/shp/shp-test_shp_50.pkl",
                "models/shp/shp-test_shp_5.pkl",
                "models/shp/shp-test_shp_14.pkl",
                "models/shp/shp-train_cpm_50.pkl",
                "models/shp/shp-train_shp_50.pkl",
                "models/shp/shp-train_shp_5.pkl",
                "models/shp/shp-train_shp_14.pkl",
            ]
        },
        "hh": {
            "data": "hh-rlhf-bon",
            "models": [
                "models/hh/hh-rlhf-test_cpm_50.pkl",
                "models/hh/hh-rlhf-test_hh_50.pkl",
                "models/hh/hh-rlhf-test_hh_5.pkl",
                "models/hh/hh-rlhf-test_hh_14.pkl",
                "models/hh/hh-rlhf-train_cpm_50.pkl",
                "models/hh/hh-rlhf-train_hh_50.pkl",
                "models/hh/hh-rlhf-train_hh_5.pkl",
                "models/hh/hh-rlhf-train_hh_14.pkl",
            ]
        }
    }
    
    for dataset_name, dataset_config in datasets.items():
        data = load_dataset('Bravansky/compositional-preference-modeling', name=dataset_config["data"])["train"].to_pandas()
        print(data.head())
        results = pd.DataFrame({
            'prompt': data['prompt'],
            'response': data['response']
        })

        for model_path in dataset_config["models"]:
            scores, model_name = load_model_scores(data, model_path)
            results[f'score_{model_name}'] = scores

        output_path = Path(f"out/{dataset_name}/results/reward_generated_results.feather")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_feather(output_path)
        print(f"Saved scores for {dataset_name} to {output_path}")
        print("\nFirst few rows:")
        print(results.head())

if __name__ == "__main__":
    main()