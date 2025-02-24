import os
import pandas as pd
import argparse
from utils.openai_generator import OpenAIGenerator
from utils.open_router_generator import OpenRouterGenerator
from datetime import datetime
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv()

system_prompts = {
    "vicuna": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    "mistral": None,
    "mixtral": None,
    "tulu2": None,
    "gpt-3.5": "You are a helpful assistant.",
    "gpt-4": "You are a helpful assistant.",
    "gpt-4o": "You are a helpful assistant.",
    "gemini": "You are a helpful assistant.",
    "llama": "You are a helpful assistant."
}

model_classes = {
    "GPT-4o": OpenAIGenerator(model_name="gpt-4o"),
    "Gemini Flash 1.5": OpenRouterGenerator(model_name="google/gemini-flash-1.5", default_system_prompt=system_prompts["gemini"], evaluation=True),
    "Llama 3.1 Instruct": OpenRouterGenerator(model_name="meta-llama/llama-3.1-8b-instruct", default_system_prompt=system_prompts["llama"], evaluation=True)
}

def process_csv_files(models):
    
    df = load_dataset("Bravansky/compact-jailbreaks", "attacks", split="train").to_pandas()[["attack_name", "attacked_model_name", "attack"]]

    use_wandb = all(os.environ.get(var) for var in ['WANDB_PROJECT', 'WANDB_USERNAME', 'WANDB_MODE'])
    
    if use_wandb:
        experiment_name = f"jailbreak-wildteaming-evaluation-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        wandb.init(
            project=os.environ['WANDB_PROJECT'],
            entity=os.environ['WANDB_USERNAME'],
            mode=os.environ['WANDB_MODE'],
            name=experiment_name
        )
        wandb.run.log_code(".")
    
    for model_name in models:
        generator = model_classes.get(model_name.strip())
        if not generator:
            print(f"Model {model_name} not found.")
            continue

        attack_prompts = df["attack"].tolist()
        attack_responses = generator.generate_responses(attack_prompts)

        df[f"{model_name}_response"] = attack_responses

        output_filename = f"jailbreak_results_{model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
        output_path = os.path.join("outputs", output_filename)
        
        df.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")
        
        if use_wandb:
            wandb.save(output_path)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    
    process_csv_files(["Gemini Flash 1.5", "GPT-4o", "Llama 3.1 Instruct"])