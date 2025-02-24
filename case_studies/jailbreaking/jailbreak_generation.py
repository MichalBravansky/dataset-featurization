import pandas as pd
import wandb
from typing import List
import random
from utils.data_utils import generate_jailbreak, create_test_prompt_dict
from datetime import datetime
import numpy as np
from utils.together_generator import TogetherGenerator
import os
import wandb
from constants import LLAMA_JAILBREAK_FEATURES, WILDTEAMING_JAILBREAK_FEATURES
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv() 

random.seed(42)

feature_names = WILDTEAMING_JAILBREAK_FEATURES
TYPE = "wildteaming"

def generate_attacks_for_behavior(df: pd.DataFrame, behavior: str, feature_count: int, num_attacks: int) -> List[str]:
    """
    Generate attacks for a given behavior with specified feature count.
    Retry only for test prompts that result in NaN responses.
    """
    
    vanilla_prompts = [behavior]
    test_prompts = create_test_prompts(df, vanilla_prompts, samples_per_prompt=num_attacks)
    
    _, responses = run_experiment(
        df=df,
        test_prompts=test_prompts,
        num_features=feature_count
    )
  
    max_retries = 5
    retry_count = 0
    while any(pd.isna(responses)) or any(response is None or response == "" for response in responses):
        if retry_count >= max_retries:
            print("Max retries reached. Some responses may still be NaN or empty.")
            break
        
        nan_indices = [i for i, response in enumerate(responses) if pd.isna(response) or response is None or response == ""]
        if not nan_indices:
            break

        new_test_prompts = [create_test_prompts(df, [vanilla_prompts[0]], samples_per_prompt=1)[0] for _ in nan_indices]
        
        _, new_responses = run_experiment(
            df=df,
            test_prompts=new_test_prompts,
            num_features=feature_count
        )

        for i, index in enumerate(nan_indices):
            responses[index] = new_responses[i]
        
        retry_count += 1
    
    return responses

def create_test_prompts(df: pd.DataFrame, vanilla_prompts: List[str], samples_per_prompt: int = 1) -> List[dict]:
    """
    Create test prompts with randomly sampled feature vectors
    """
    test_prompts = []
    feature_cols = [col for col in df.columns if col.startswith("The selected string")]
    
    for prompt in vanilla_prompts:
        for i in range(samples_per_prompt):

            random_row = df.sample(n=1).iloc[0]
            test_prompts.append(
                create_test_prompt_dict(
                    prompt=prompt,
                    features=feature_cols,
                    feature_vector=[bool(val) for val in random_row[feature_cols].values.tolist()]
                )
            )
    
    return test_prompts

def run_experiment(df: pd.DataFrame, test_prompts: List[dict], num_features: int):
    """
    Run experiment with specified number of features and return both generated and vanilla prompts
    """

    selected_features = feature_names[:num_features]
    generated_prompts = generate_jailbreak(
        df=df,
        selected_features=selected_features,
        test_prompts=test_prompts,
        use_features=True
    )
    
    generator = TogetherGenerator(
        max_retries=3,
        retry_delay=2,
        max_workers=15
    )

    responses = generator.generate_responses(generated_prompts)
    
    result_pairs = list(zip(
        [p["vanilla"] for p in test_prompts],
        generated_prompts,
        [p["feature_vector"] for p in test_prompts]
    ))
    
    return result_pairs, responses

def main():
    os.makedirs("outputs", exist_ok=True)

    use_wandb = all(os.environ.get(var) for var in ['WANDB_PROJECT', 'WANDB_USERNAME', 'WANDB_MODE'])
    
    if use_wandb:
        experiment_name = f"jailbreak-creation-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        wandb.init(
            project=os.environ['WANDB_PROJECT'],
            entity=os.environ['WANDB_USERNAME'],
            mode=os.environ['WANDB_MODE'],
            name=experiment_name
        )
        
    df = load_dataset("Bravansky/compact-jailbreaks", TYPE, split="train").to_pandas()
    verified_df = load_dataset("Bravansky/compact-jailbreaks", TYPE + "-evaluation", split="train").to_pandas()
    df = df.merge(verified_df, left_on="adversarial", right_on="string")

    vanilla_prompts = load_dataset("Bravansky/compact-jailbreaks", "harmbench", split="train").to_pandas()["Behavior"].tolist()

    test_prompts = create_test_prompts(df, vanilla_prompts, samples_per_prompt=30)

    feature_counts = [int(x) for x in [5,10,15,20,25,30,35,40,45,50]]
    
    for num_features in feature_counts:
        print(f"Running experiment with {num_features} features...")
        
        prompt_pairs, responses = run_experiment(
            df=df,
            test_prompts=test_prompts,
            num_features=num_features
        )

        max_retries = 10
        retry_count = 0
        while any(pd.isna(responses)) or any(response is None or response == "" for response in responses):
            if retry_count >= max_retries:
                print("Max retries reached. Some responses may still be NaN or empty.")
                break
            
            nan_indices = [i for i, response in enumerate(responses) if pd.isna(response) or response is None or response == ""]
            if not nan_indices:
                break

            valid_nan_indices = [i for i in nan_indices]

            new_test_prompts = [create_test_prompts(df, [prompt_pairs[i][0]], samples_per_prompt=1)[0] for i in valid_nan_indices]
            
            _, new_responses = run_experiment(
                df=df,
                test_prompts=new_test_prompts,
                num_features=num_features
            )
            
            for i, index in enumerate(valid_nan_indices):
                responses[index] = new_responses[i]
            
            retry_count += 1
        
        vanilla_prompts, generated_prompts, feature_vectors = zip(*prompt_pairs)
        
        if use_wandb:
            wandb.log({
                "num_features": num_features,
                "generated_prompts": generated_prompts,
                "vanilla_prompts": vanilla_prompts,
                "responses": responses,
                "num_test_prompts": len(test_prompts)
            })
        
        results_df = pd.DataFrame({
            "vanilla_prompt": vanilla_prompts,
            "generated_prompt": generated_prompts,
            "model_response": responses,
            "feature_vector": feature_vectors
        })

        output_file = f"outputs/jailbreak_creation_{num_features}.csv"
        results_df.to_csv(output_file, index=False)
        if use_wandb:
            wandb.save(output_file)

if __name__ == "__main__":
    main() 