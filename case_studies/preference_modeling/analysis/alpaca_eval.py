import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from datasets import load_dataset
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import random

TOKEN = ""
DATASET_TYPE = "" # "shp" or "hh"
DATASET_NAME = "" # e.g. "vanilla_score_vs_shp-with-features-results_train_cpm_50.pkl.feather"

def find_score(score_list):
    result_list = []
    for gen_sentence in score_list:
        search1 = re.findall(r'^\s*([\d.]+)\s+([\d.]+)\s*$', gen_sentence, re.MULTILINE)
        search2 = re.findall(r'(?:Assistant 1:|Assistant 2:)\s+(\d+)', gen_sentence)
        if len(search1)>0:
            result_list.append(' '.join(search1[0]))
        elif len(search2)>0:
            result_list.append(' '.join(search2))
        else:
            result_list.append('0 0')
    return result_list

URL = 'https://api.openai.com/v1/chat/completions'
body = {
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": None}],
    "max_tokens": 1,
    "temperature": 0,
    "logprobs": True,
    "top_logprobs": 20,
    "seed": 0
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}


alpaca_system_prompt_format = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""

alpaca_user_prompt_format = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": \"""{instruction}\"""
}}
## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": \"""{output_1}\"""
    }},
    {{
        "model_identifier": "M",
        "output": \"""{output_2}\"""
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier"""

def process_single_evaluation(args: Tuple[str, str, str, int, str]) -> Tuple[str, float, float]:
    instruction, output_1, output_2, _, prompt_format = args

    prompt = prompt_format.format(instruction=instruction, output_1=output_1, output_2=output_2)
    
    body_copy = body.copy()
    body_copy.update({'messages': [
        {"role": "system", "content": alpaca_system_prompt_format}, 
        {"role": "user", "content": prompt}
    ]})
    
    try:
        response = requests.post(URL, headers=headers, json=body_copy)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_data = response.json()

        content_logprobs = response_data['choices'][0]['logprobs']["content"][0]["top_logprobs"]
        m_logprob = next((token["logprob"] for token in content_logprobs if token["token"].strip() == 'm'), float('-inf'))
        M_logprob = next((token["logprob"] for token in content_logprobs if token["token"].strip() == 'M'), float('-inf'))
        overall_logprob = m_logprob - M_logprob

        # Swap outputs and make second request
        prompt_swap = prompt_format.format(instruction=instruction, output_1=output_2, output_2=output_1)
        body_copy.update({'messages': [
            {"role": "system", "content": alpaca_system_prompt_format},
            {"role": "user", "content": prompt_swap}
        ]})
        response_swap = requests.post(URL, headers=headers, json=body_copy)
        response_swap.raise_for_status()
        response_swap_data = response_swap.json()

        content_logprobs = response_swap_data['choices'][0]['logprobs']["content"][0]["top_logprobs"]
        m_logprob_swap = next((token["logprob"] for token in content_logprobs if token["token"].strip() == 'm'), float('-inf'))
        M_logprob_swap = next((token["logprob"] for token in content_logprobs if token["token"].strip() == 'M'), float('-inf'))
        overall_logprob += M_logprob_swap - m_logprob_swap

        return instruction, overall_logprob > 0, overall_logprob
        
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return instruction, False, 0.0

def evaluate_parallel(dataset, args, prompt_format):
    tasks = []
    for idx in range(len(dataset)):
        instruction = dataset['prompt'][idx].replace("\n\nHuman: ", '').replace("\n\nAssistant: ", '').replace("\n\nPOST: ", '').replace("\n\nResponse: ", '')
        output_1 = dataset['response'][idx][args.indices[0]]
        output_2 = dataset['response'][idx][args.indices[1]]
        tasks.append((instruction, output_1, output_2, idx, prompt_format))
    
    tasks = tasks
    prompt_list, response_list, logprobs_pairs = [], [], []
    original_prompt_list, original_response1_list, original_response2_list = [], [], []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {executor.submit(process_single_evaluation, task): task[3] 
                        for task in tasks}
        
        results = [None] * len(dataset)
        
        for future in tqdm(as_completed(future_to_idx), total=len(dataset)):
            idx = future_to_idx[future]
            prompt, response, logprobs = future.result()
            results[idx] = (prompt, response, logprobs)
            
            original_prompt_list.append(dataset['prompt'][idx])
            original_response1_list.append(dataset['response'][idx][args.indices[0]])
            original_response2_list.append(dataset['response'][idx][args.indices[1]])
    
    for result in results:
        prompt, response, logprobs = result
        prompt_list.append(prompt)
        response_list.append(response)
        logprobs_pairs.append(logprobs)
    
    return (prompt_list, response_list, logprobs_pairs,
            original_prompt_list, original_response1_list, original_response2_list)

def main():
    class Args:
        def __init__(self):
                    pass
            
    args = Args()
    args.dataset = f'out/{DATASET_TYPE}/results/{DATASET_NAME}'
    args.indices = [0, 1]
    
    assert len(args.indices)==2,' It should be pairwise comparision'
    identifier = '_'.join([str(idx) for idx in args.indices])
    dataset = pd.read_feather(args.dataset)
    
    dataset_name = args.dataset.split('/')[-1]
    save_fname = f'./data/{dataset_name}_claude_evaluation{identifier}.json'
    if os.path.exists(save_fname):
        raise FileExistsError(f"'{save_fname}' already exists.")    

    (prompt_list, response_list, logprobs_pairs,
     original_prompt_list, original_response1_list, original_response2_list) = evaluate_parallel(dataset, args, alpaca_user_prompt_format)

    final_list = {
        'prompt': prompt_list,
        'response': response_list,
        'logprobs_pairs': logprobs_pairs,
        'original_prompt': original_prompt_list,
        'original_response1': original_response1_list,
        'original_response2': original_response2_list
    }

    with open(save_fname, encoding="utf-8", mode="w") as file:
        json.dump(final_list, file)
    print(f"Saved to {save_fname}")

main()