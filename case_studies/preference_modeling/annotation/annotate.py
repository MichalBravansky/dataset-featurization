from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
import sys
sys.path.append('../')
from constants import hh_constants, shp_constants, cpm_constants
from common import get_lm, feature_score, get_feature_extractor
from functools import partial
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import random
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

hh_constants = {f"hh-{i}": constant for i, constant in enumerate(hh_constants)}
shp_constants = {f"shp-{i}": constant for i, constant in enumerate(shp_constants)}
cpm_constants = {f"cpm-{i}": constant for i, constant in enumerate(cpm_constants)}


def process_row(row, feature_extractor, data_type, features):
    prompt = row['prompt']
    for prefix in ['\n\nHuman: ', '\n\nAssistant: ', '\n\nPOST: ', '\n\nResponse: ']:
        prompt = prompt.replace(prefix, '')
        
    if isinstance(row['response'], list):
        score_lists = []
        length_scores = []

        feature_names = list(features.keys())
        shuffled_feature_names = feature_names.copy()
        random.shuffle(shuffled_feature_names)
        shuffled_features = {name: features[name] for name in shuffled_feature_names}
        
        for response in row['response']:
            scores = feature_score(OpenAI(), data_type=data_type, features=shuffled_features, history=prompt, reply=response)
            length_scores.append(len(response) / 100)
            score_lists.append(scores)

        for i, feature_name in enumerate(shuffled_feature_names):
            row[f'zeroshot_{feature_name}'] = [scores[i] for scores in score_lists]
            
        row['zeroshot_length'] = length_scores
            
    else:
        feature_names = list(features.keys())
        shuffled_feature_names = feature_names.copy()
        random.shuffle(shuffled_feature_names)
        shuffled_features = {name: features[name] for name in shuffled_feature_names}
        
        scores = feature_score(OpenAI(), data_type=data_type, features=shuffled_features, history=prompt, reply=row['response'])
        length_score = len(row['response']) / 100

        for i, feature_name in enumerate(shuffled_feature_names):
            row[f'zeroshot_{feature_name}'] = scores[i]
            
        row['zeroshot_length'] = length_score
            
    return row

def process_dataset(dataset, file_name, options, context, response_prefix, features, split=None, eval_generated=False, model_name='gpt-3.5-turbo', model_type='openai'):
    lm = get_lm(model_name, model_type)
    feature_extractor = get_feature_extractor(lm, data_type='hh' if 'hh' in dataset else 'shp')

    dataset_obj = load_dataset('Bravansky/compositional-preference-modeling', name=dataset, split=split or "train")
    df = pd.DataFrame(dataset_obj)
   
    if eval_generated:
        pass
    else:
        if 'hh' in dataset:
            df['prompt'] = df['human']
            df['response'] = df[['assistant_chosen', 'assistant_rejected']].values.tolist()
        else:
            df['prompt'] = df['history']
            df['response'] = df[['human_ref_A', 'human_ref_B']].values.tolist()

    process_func = partial(process_row, feature_extractor=feature_extractor, data_type='hh' if 'hh' in dataset else 'shp', features=features)
 
    with ThreadPoolExecutor(max_workers=10) as executor:
        rows = df.to_dict('records')
        processed_rows = list(tqdm(executor.map(process_func, rows), total=len(rows), desc="Processing rows"))
  
    df_processed = pd.DataFrame(processed_rows)

    save_fname = f'{dataset}-{split if split else "full"}'
    if eval_generated:
        save_fname += '-bon'

    df_processed.to_parquet(f'outputs/{save_fname}-results.parquet', index=False)
    print(f"Saved to {save_fname}-results.parquet")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='gpt-4o-2024-08-06')
    ap.add_argument('--model-type', default='openai')
    args = ap.parse_args()

    datasets = [
        {
            'dataset': 'hh-rlhf-bon',
            'options': None,
            'context': None,
            'response_prefix': None,
            'eval_generated': True,
            'split': False
        },
        {
            'dataset': 'shp-bon',
            'options': None,
            'context': None,
            'response_prefix': None,
            'eval_generated': True,
            'split': False
        },
        {
            'dataset': 'hh-rlhf',
            'options': ['chosen', 'rejected'],
            'context': 'human',
            'response_prefix': 'assistant',
            'eval_generated': False,
            'split': 'train'
        },
        {
            'dataset': 'hh-rlhf',
            'options': ['chosen', 'rejected'],
            'context': 'human',
            'response_prefix': 'assistant',
            'eval_generated': False,
            'split': 'test'
        },
        {
            'dataset': 'shp',
            'options': ['A', 'B'],
            'context': 'history',
            'response_prefix': 'human_ref',
            'eval_generated': False,
            'split': 'train'
        },
        {
            'dataset': 'shp',
            'options': ['A', 'B'],
            'context': 'history',
            'response_prefix': 'human_ref',
            'eval_generated': False,
            'split': 'test'
        }
    ]

    # Process all datasets
    for dataset_config in datasets:
        if dataset_config['eval_generated']:
            process_dataset(
                dataset=dataset_config['dataset'],
                file_name=dataset_config['dataset'],
                options=dataset_config['options'],
                context=dataset_config['context'],
                response_prefix=dataset_config['response_prefix'],
                split=dataset_config['split'],
                eval_generated=True,
                model_name=args.model_name,
                model_type=args.model_type,
                features = {**cpm_constants, **(hh_constants if 'hh' in dataset_config['dataset'] else shp_constants)}
            )
        else:
            process_dataset(
                dataset=dataset_config['dataset'],
                file_name=dataset_config['dataset'],
                options=dataset_config['options'],
                context=dataset_config['context'],
                response_prefix=dataset_config['response_prefix'],
                split=dataset_config['split'],
                eval_generated=False,
                model_name=args.model_name,
                model_type=args.model_type,
                features = {**cpm_constants, **(hh_constants if 'hh' in dataset_config['dataset'] else shp_constants)}
            )

if __name__ == "__main__":
    main()