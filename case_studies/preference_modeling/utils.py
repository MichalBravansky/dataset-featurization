# This is a fork of shp4.py which uses LLL models instead of logistic-regression ones
from tqdm import tqdm
import dill as pickle
from transformers import AutoModelForSeq2SeqLM
import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np
from pathlib import Path
import argparse
import streamlit as st

def load_scored_samples(scored_samples):
    if scored_samples.exists():
        return pd.read_json(scored_samples)
    else:
        return datasets.load_dataset(str(scored_samples))['train'].to_pandas()

def add_callback_features(scored_samples, feature_callbacks):
    for feature_name, feature_fn in feature_callbacks.items():
        scored_samples[feature_name] = [feature_fn(response, prompt) for response, prompt in zip(scored_samples['response'], scored_samples['prompt'])]
    return scored_samples

def features_values(scored_samples, feature_names):
    return scored_samples[feature_names].to_numpy()

def add_model_scores(label, scored_samples, model, features_values_matrix, standardize=True):
    scored_samples[label] = model_scores(model, features_values_matrix)
    if standardize:
        if 'reward_mean' in model and 'reward_std' in model:
            scored_samples[label] -= model['reward_mean']
            scored_samples[label] /= model['reward_std']
        else:
            scored_samples[label] -= scored_samples[label].mean()
            scored_samples[label] /= scored_samples[label].std()
    return scored_samples

def model_scores(model, feature_values):
    return (model['coefficients'] * (feature_values - model['features_mean']) / model['features_std']).sum(1)


def best_of_n(scored_samples):
    bon_data = []
    for prompt, scored_prompt_samples in scored_samples.groupby('prompt'):
        for n in [1, 2, 4, 8, 16]:
            best_of_A = np.argmax(scored_prompt_samples['score_A'][:n])
            best_of_A_score_A = scored_prompt_samples['score_A'].iloc[best_of_A]
            best_of_A_score_B = scored_prompt_samples['score_B'].iloc[best_of_A]
            response = scored_prompt_samples['response'].iloc[best_of_A]
            bon_data.append({'n': n, 'best_of_A_score_A': best_of_A_score_A,
                                        'best_of_A_score_B': best_of_A_score_B, 'response': response,
                                        'prompt': prompt})
    return pd.DataFrame(bon_data, columns=['n', 'best_of_A_score_A', 'best_of_A_score_B', 'response', 'prompt'])

def plot_corr(scored_samples):
    plt.clf()
    sns.set_theme('paper')
    all_scores_A = scored_samples['score_A']
    all_scores_B = scored_samples['score_B']
    fig, ax = plt.subplots()
    plt.scatter(all_scores_A, all_scores_B)
    plt.xlabel('Score according to PM A')
    plt.ylabel('Score according to PM B')
    plt.show()
    return fig

def corr(scored_samples):
    all_scores_A = scored_samples['score_A']
    all_scores_B = scored_samples['score_B']
    return pearsonr(all_scores_A, all_scores_B)

def plot_data(bon_df):
    plt.clf()
    fig, ax = plt.subplots()
    sns.lineplot(data=bon_df, x='n', y='best_of_A_score_A', ax=ax, legend='full', label='PM A (used for argmax)')
    sns.lineplot(data=bon_df, x='n', y='best_of_A_score_B', ax=ax, legend='full', label='PM B')
    ax.set_xlabel('n')
    ax.set_ylabel('Score')
    return fig