import os
import sys
import pandas as pd
import ast
import statistics
import wandb
from datetime import datetime
from config import WANDB_PROJECT, WANDB_USERNAME, WANDB_MODE, FEATURE_FILTRATION_THRESHOLD
from utils.generator import Generator
from utils.verifier import Verifier
from utils.filtration import Filter
from utils.perplexity_evaluator import Evaluator
from utils.data_loader import DataLoader
import argparse

def main(experiment_name, dataset_name, output_dir, num_iterations, batch_size):
    """
    Main function for feature extraction and evaluation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if WANDB_PROJECT:
        experiment_name = f"{experiment_name}-featurization-{dataset_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        wandb.init(project=WANDB_PROJECT, entity=WANDB_USERNAME, mode=WANDB_MODE, name=experiment_name)
        wandb.run.log_code(".")

    evaluator = Evaluator(batch_size=batch_size)

    if DataLoader.is_supported_dataset(dataset_name):
        df = DataLoader.get_dataset(dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")

    df.reset_index(drop=True, inplace=True)
        
    evaluator.init_cached_prompts(df)

    best_features = []
    df = df[[column for column in df.columns if "_property" not in column]]
    losses = []

    best_features_file = os.path.join(output_dir, f"best_features.txt")
    with open(best_features_file, "w") as bf_file:
        bf_file.write("")
    
    with open(os.path.join(output_dir, f"perplexities.txt"), "w") as losses_file:
        losses_file.write("")

    if WANDB_PROJECT:
        wandb.save(best_features_file)
        wandb.save(os.path.join(output_dir, f"perplexities.txt"))

    verified_df = load_verified_df(df, dataset_name)

    for _ in range(num_iterations):
        if best_features:
            best_feature = best_features[-1]
            sub_df = df[df[best_feature + "_property"]]
            temp_evaluated_df = evaluator.evaluate(
                verified_df[verified_df["string"].isin(sub_df["string"])].reset_index(drop=True),
                sub_df.reset_index(drop=True),
                list(sub_df.index),
                feature_names=best_features
            )
            temp_evaluated_df.index = verified_df[verified_df["string"].isin(sub_df["string"])].index
            evaluated_df.loc[verified_df[verified_df["string"].isin(sub_df["string"])].index] = temp_evaluated_df
        else:
            evaluated_df = evaluator.evaluate(verified_df, df, list(df.index), feature_names=best_features)

        loss = statistics.mean(evaluated_df["empty"].to_list())
        losses.append(evaluated_df["empty"].to_list())

        best_feature = select_best_feature(evaluated_df, best_features)

        if best_feature == "empty":
            sys.exit("Terminating the program because no additional features that lower perplexity can be found.")

        df[best_feature + "_property"] = verified_df[best_feature]
        best_features.append(best_feature)

        log_results(best_feature, evaluated_df, best_features, best_features_file, loss, output_dir, WANDB_PROJECT is not None)


def load_verified_df(df, dataset_name):
    """
    Load and process the verified features DataFrame
    
    Args:
        df: DataFrame containing original strings
        input_dir: Directory containing evaluation data
        
    Returns:
        DataFrame: Filtered evaluation data containing only features above threshold
    """
    verified_df = DataLoader.get_evaluation(dataset_name)
    verified_df.reset_index(drop=True, inplace=True)
    
    verified_df = pd.merge(df["string"], verified_df.drop_duplicates(subset=["string"]), on="string", how="left")
    verified_df = verified_df[list(verified_df.columns)[1:] + ["string"]]

    column_sums = verified_df.drop(columns=["string"]).sum(axis=0)
    threshold_sum = len(verified_df) * FEATURE_FILTRATION_THRESHOLD
    selected_by_sum = column_sums[column_sums >= threshold_sum].index.tolist() + ["string"]

    return verified_df.loc[:, selected_by_sum]


def select_best_feature(evaluated_df, best_features):
    """
    Select the best feature based on evaluation scores
    """
    remaining_features = [col for col in evaluated_df.columns if col not in best_features]
    feature_scores = evaluated_df[remaining_features].sum().sort_values()
    return feature_scores.index[0]


def log_results(best_feature, evaluated_df, best_features, best_features_file, loss, output_dir, use_wandb = False):
    """
    Log results to files and wandb
    """
    print(f"Selected feature: {best_feature}")

    remaining_features = [col for col in evaluated_df.columns[:-1] if col not in best_features]
    remaining_scores = evaluated_df[remaining_features].sum().sort_values()
    print(f"Top 5 remaining features: {remaining_scores.index[:5].tolist()}")

    with open(best_features_file, "w") as bf_file:
        bf_file.write("\n".join(best_features) + "\n")

    with open(os.path.join(output_dir, f"perplexities.txt"), "a") as losses_file:
        losses_file.write(f"{loss}\n")
    
    if use_wandb:
        wandb.log({
            "mean_perplexity": loss
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features from a dataset.")
    parser.add_argument("experiment_name", nargs='?', type=str, default="example",
                      help="Name of the experiment to log into wandb")
    parser.add_argument("dataset_name", nargs='?', type=str, default="example_dataset",
                      help="Name of the dataset to analyze")
    parser.add_argument("output_dir", nargs='?', type=str, default="data",
                      help="Directory to save the produced features")
    parser.add_argument("num_iterations", nargs='?', type=int, default=10,
                      help="Number of features to produce")
    parser.add_argument("batch_size", nargs='?', type=int, default=16,
                      help="Batch size for featurization")
    args = parser.parse_args()

    main(args.experiment_name, args.dataset_name, args.output_dir,
         args.num_iterations, args.batch_size)