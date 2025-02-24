import os
import sys
import pandas as pd
import ast
import argparse
from config import WANDB_PROJECT, WANDB_USERNAME, WANDB_MODE
from utils.generator import Generator
from utils.verifier import Verifier
from utils.filtration import Filter
import wandb
from datetime import datetime
from datasets import load_dataset
from utils.data_loader import DataLoader

def main(dataset_name, output_dir):
    if WANDB_PROJECT:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_USERNAME, mode=WANDB_MODE, name = f"modeling-{type}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}")

    os.makedirs(output_dir, exist_ok=True)

    generator = Generator()
    verifier = Verifier()
    filtration = Filter()

    if DataLoader.is_supported_dataset(dataset_name):
        df = DataLoader.get_dataset(dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    
    df.reset_index(drop=True, inplace=True)
    
    generated_df = pd.DataFrame()
    generated_df["string"] = df.apply(lambda x: str(x["string"]), axis=1).values

    features = generator.analyze(generated_df)

    features_file_path = os.path.join(output_dir, f"generated_features.txt")
    with open(features_file_path, "w") as file:
        file.write("\n".join(features))
        
    if WANDB_PROJECT:
        wandb.save(features_file_path)

    filtered_features = filtration.filter(features)

    filtered_features_file_path = os.path.join(output_dir, f"clustered_features.txt")
    with open(filtered_features_file_path, "w") as file:
        file.write("\n".join(filtered_features))
        
    if WANDB_PROJECT:
        wandb.save(filtered_features_file_path)

    verified_df = verifier.process(df["string"].to_list(), filtered_features)
    verified_df_path = os.path.join(output_dir, f"evaluation_df.csv")
    verified_df.to_csv(verified_df_path, index=False)
    
    if WANDB_PROJECT:
        wandb.save(verified_df_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate feature propositions for a dataset.")
    parser.add_argument("output_dir", nargs='?', type=str, help="Directory to save the features produced by the system", default="data")
    parser.add_argument("dataset_name", nargs='?', type=str, help="Name of the dataset with the strings to be analyzed", default="example_dataset")
    args = parser.parse_args()

    main(args.dataset_name, args.output_dir)