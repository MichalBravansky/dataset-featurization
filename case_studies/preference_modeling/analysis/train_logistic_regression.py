import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV
import numpy as np
import re
import random
from pathlib import Path
import dill as pickle
import warnings
warnings.filterwarnings("ignore")
from datasets import Dataset, DatasetDict
import ast
import sys
from constants import cpm_constants, hh_constants, shp_constants

hh_constants = {f"hh-{i}": constant for i, constant in enumerate(hh_constants)}
shp_constants = {f"shp-{i}": constant for i, constant in enumerate(shp_constants)}
cpm_constants = {f"cpm-{i}": constant for i, constant in enumerate(cpm_constants)}

FEATURE_CUTS = [5, 14, 50]

def train_model_g(dataset, model_name, feature_names):

    feature_cols = feature_names
    feature_values = np.array(dataset[feature_cols].values)
    

    feature_values_A = np.array([[float(cell[0]) for cell in row] for row in feature_values])
    feature_values_B = np.array([[float(cell[1]) for cell in row] for row in feature_values])
        
    concat_feature_values = np.concatenate([feature_values_A,feature_values_B])
    feature_means = concat_feature_values.mean(0)
    feature_stds = concat_feature_values.std(0)+1e-7
    concat_feature_values -= feature_means
    concat_feature_values /= (feature_stds)
    feature_values_A = concat_feature_values[:len(feature_values)]
    feature_values_B = concat_feature_values[len(feature_values):]
    feature_values = (feature_values_A-feature_values_B)
    
    pipeline = Pipeline(steps=[('classifier', LogisticRegression(fit_intercept=False,random_state=42))])
    param_grid = {
        'classifier__penalty': ['l1','l2'],
        'classifier__C': np.logspace(-5, 5, 12),
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=8, scoring='accuracy', n_jobs=-1)
    grid_search.fit(feature_values, dataset['labels'])
    print(f"{model_name} Best hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    print(f"{model_name}  accuracy: {best_accuracy}")
    best_model.mean = feature_means
    best_model.std = feature_stds
    return best_model,best_accuracy

def train_model_f(dataset, model_name, feature_names):
    features = [f'{feature_name}_A' for feature_name in feature_names] + [f'{feature_name}_B' for feature_name in feature_names]
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression(fit_intercept=False,random_state=42))])
    param_grid = {
        'classifier__penalty': ['l1','l2'],
        'classifier__C': np.logspace(-5, 5, 12),
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=8, scoring='accuracy', n_jobs=-1)
    grid_search.fit(dataset[features], dataset['labels'])
    print(f"{model_name} Best hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    print(f"{model_name}  accuracy: {best_accuracy}")
    # joblib.dump(best_model, f'{model_name}.joblib')
    return best_model,best_accuracy

def recalibrate_and_scores(values, weights, means, std):
    normalized_values = (values - means) / std
    return np.dot(normalized_values, weights)

def get_model_statistics(model,ds, feature_names):
    try:
        # case f
        weights = np.mean([model.named_steps.classifier.coef_[0,:len(feature_names)],-model.named_steps.classifier.coef_[0,len(feature_names):]],axis=0)
        means = model.named_steps.scaler.mean_.reshape(2,-1).T.mean(1)
        stds = model.named_steps.scaler.scale_.reshape(2,-1).T.mean(1)
    except:
        # case g
        weights = model.named_steps.classifier.coef_.flatten()
        means = model.mean
        stds = model.std
    
    print('Model weights:', weights)
    print('Model means:', means)
    print('Model stds:w', stds)
    
    # inference
    inference_dict = dict()
    for inference_subset in ['train','test']:
        inference_dict[inference_subset] = {'chosen':[],'rejected':[]}

        df = ds[inference_subset].to_pandas()
        labels = df['labels'].values
        
        feature_values = df[feature_names].values

        feature_values_A = np.array([[float(cell[0]) for cell in row] for row in feature_values])
        feature_values_B = np.array([[float(cell[1]) for cell in row] for row in feature_values])

        option1_score = recalibrate_and_scores(feature_values_A, weights, means, stds)
        option2_score = recalibrate_and_scores(feature_values_B, weights, means, stds)

        for score1, score2, label in zip(option1_score,option2_score,labels):
            chosen = score1 if label==1 else score2
            rejected = score2 if label==1 else score1
            inference_dict[inference_subset]['chosen'].append(chosen)
            inference_dict[inference_subset]['rejected'].append(rejected)

        inference_dict[inference_subset]['option1']=option1_score
        inference_dict[inference_subset]['option2']=option2_score

    print("Train accuracy: ",np.mean(np.array(inference_dict['train']['chosen'])>np.array(inference_dict['train']['rejected'])))

    print("Test accuracy: ",np.mean(np.array(inference_dict['test']['chosen'])>np.array(inference_dict['test']['rejected'])))
    
    return weights,means,stds,inference_dict, np.mean(np.array(inference_dict['test']['chosen'])>np.array(inference_dict['test']['rejected']))

def preprocess_hh_dataset(example,options=['A','B']):
    chosen_col = [x for x in example.keys() if "hh" in x or 'cpm' in x]
    
    if random.random()<0.5:
        # chosen -> A
        example['labels']=1
        
        for col in chosen_col:
            example[col] = [example[col][0], example[col][1]]
    else:
        # chosen -> B
        example['labels']=0
        for col in chosen_col:    
            example[col] = [example[col][1], example[col][0]]
            
    return example


for original_train_dataset, original_test_dataset in [["data/hh-rlhf-results.parquet", "data/hh-rlhf-test-results.parquet"], ["data/shp-with-features-results.parquet", "data/shp-with-features-test-results.parquet"]]:

    for feature_type in ["cpm", "other"]:
        
        for dataset_split in ["train", "test"]:
        
            if feature_type == "cpm":
                feature_cut_lengths = [50]
            else:
                feature_cut_lengths = FEATURE_CUTS
                
            for feature_cut_length in feature_cut_lengths:
            
                if dataset_split == "test":
                    train_dataset = original_test_dataset
                    test_dataset = original_train_dataset
                else:
                    train_dataset = original_train_dataset
                    test_dataset = original_test_dataset
                    
                train_df = pd.read_parquet(train_dataset)
                test_df = pd.read_parquet(test_dataset)

                test_df.reset_index(drop=True, inplace=True)
                train_df.reset_index(drop=True, inplace=True)
            
                ds = DatasetDict()
                ds['train'] = Dataset.from_pandas(train_df)
                ds['test'] = Dataset.from_pandas(test_df)
                
                if feature_type == "other":
                    feature_type = "hh" if "hh" in train_dataset else "shp"
                    
                FEATURES = [col for col in train_df.columns.tolist() if feature_type in col]
                
                # Removing NaN values
                for split in ['train', 'test']:
                    df = ds[split].to_pandas()

                    mask = pd.DataFrame([
                        [not (pd.isna(row[feat][0]) or pd.isna(row[feat][1])) for feat in FEATURES]
                        for _, row in df.iterrows()
                    ]).all(axis=1)
                    df = df[mask].reset_index(drop=True)
                    ds[split] = Dataset.from_pandas(df)
                    
                differences_std = {}
                for feat in FEATURES:
                    differences = np.array([row[feat][0] - row[feat][1] for id, row in df.iterrows()])
                    differences_std[feat] = np.std(differences)
                
                if feature_type != "cpm":
                    FEATURES = [feat for feat in FEATURES if differences_std[feat] >= 1]
                    FEATURES = FEATURES[:feature_cut_length]
                    
                    FEATURES = [feature for feature in FEATURES if "zeroshot_shp-50" != feature and "zeroshot_hh-50" != feature]
                        
                        
            
                class Args:
                    def __init__(self):
                        pass
                
                args = Args()
                args.dataset_type = train_dataset
                args.g_fn = True 
                args.shp_hh_dataset_type = train_dataset
                args.split = "train"
                args.subset_fit = 0

                suffix = ''
                    
                ################ Data Load ################
                ds_list = [ds]

                if 'hh' in args.dataset_type:
                    args.context='human'
                    args.response_prefix='assistant'
                elif 'shp' in args.dataset_type:
                    args.context = 'history'
                    args.response_prefix='human_ref'
                feature_names = FEATURES

                train_model = train_model_f 
                if args.g_fn:
                    train_model = train_model_g
                    suffix += '_g'
                    
                    

                short_data_name = args.dataset_type.split('/')[-1] + ('_concated_ds' if len(args.shp_hh_dataset_type)>1 else '')
                if 'hh' in args.dataset_type:
                    ds = ds.map(preprocess_hh_dataset)

                
                if args.subset_fit>0:
                    suffix += f'_subset_{args.subset_fit}'
                    ds['train'] = ds['train'].shuffle(seed=100).select(range(args.subset_fit))
                    ds['test'] = ds['test'].shuffle(seed=100).select(range(args.subset_fit))      
                    

                if 'human_ref_A' in ds['train'].features:
                    LENGTH_CUT = 1e8
                    print('Apply Filter')
                    ds = ds.filter(lambda x: len(x["human_ref_A"]) < LENGTH_CUT and len(x["human_ref_B"]) < LENGTH_CUT)

                ds = ds.remove_columns([feature for feature in ds['train'].features if any([targ_feat in feature for targ_feat in feature_names+['labels']]) is False])
                    
                ################ Fit Model ################
                
                gold_model,model_acc = train_model(ds[args.split].to_pandas(), args.split, feature_names)
                
                print(f"Gold model accuracy {train_dataset}-{feature_type}: ", model_acc)

                weights,means,stds,inference_dict, test_acc =get_model_statistics(gold_model, ds, feature_names)
                rewards = np.concatenate([inference_dict[args.split]['chosen'],inference_dict[args.split]['rejected']])

                train_size = len(ds["train"])
                output_dir = Path(f'./out/{train_size}')
                output_dir.mkdir(parents=True, exist_ok=True)

                save_fname = f'{short_data_name}_{args.split}_{feature_cut_length}_logistic{suffix}'

                output_path = Path(save_fname).stem+f'_{args.split}_{feature_type}_{feature_cut_length}.pkl'
                pickle.dump(
                        {'feature_names': feature_names,
                            'features_mean': means,
                            'features_std': stds,
                            'coefficients': weights,
                            "reward_mean": rewards.mean(),
                            "reward_std": rewards.std(),         
                            'acc':model_acc},
                        open(output_dir / output_path, 'wb'))
                print(f'saved to {output_dir / output_path}')
                
                with open(output_dir / 'accuracies.txt', 'a') as f:
                    f.write(f"{train_dataset}-{feature_type}-{feature_cut_length}: {test_acc}\n")

                print('Finish without making second proxy model')