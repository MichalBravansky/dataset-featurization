from typing import List, Dict, Union
import os
import pandas as pd
import datasets
from config import SUPPORTED_DATASETS

class DataLoader:
    """Handles loading and validation of datasets for feature generation"""

    @classmethod
    def load_data(cls, dataset_path_or_name: str, config = None) -> pd.DataFrame:
        """
        Load data from either a local file or remote dataset
        
        Args:
            dataset_path_or_name: Path to local file or name of remote dataset
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        if os.path.isfile(dataset_path_or_name):
            return datasets.load_dataset(
                "csv",
                config,
                data_files=dataset_path_or_name,
                split="train"
            )

        return datasets.load_dataset(dataset_path_or_name, config, split="train")

    @classmethod
    def is_supported_dataset(cls, dataset_name: str) -> bool:
        """Check if the dataset is supported"""
        if "experiments_" in dataset_name:
            return True
        return dataset_name in SUPPORTED_DATASETS

    @classmethod
    def get_dataset(cls, dataset_name: str) -> pd.DataFrame:
        """
        Load and process dataset texts
        
        Args:
            dataset_name: Name of the dataset to load or path to local file
            
        Returns:
            DataFrame with 'string' column containing processed texts
            
        Raises:
            NotImplementedError: If dataset is not supported
        """

        if not cls.is_supported_dataset(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} is not supported")

        if "experiments_" in dataset_name:
            dataset_type, group = dataset_name.split('_')[1:]
            df = datasets.load_dataset("Bravansky/dataset-featurization", dataset_type, "train").to_pandas()
            df = df[df["group"] == int(group)]
            df["string"] = df["text"]
            return df.reset_index(drop=True)

        dataset_config = SUPPORTED_DATASETS[dataset_name]
        df = DataLoader.load_data(
            dataset_config["source"],
            dataset_config.get("config", None)
        ).to_pandas()
        
        df["string"] = df[dataset_config["text_column"]]
        return df.reset_index(drop=True)

    @classmethod
    def get_evaluation(cls, dataset_name: str) -> pd.DataFrame:
        """
        Load evaluation dataset for featurization
        
        Args:
            dataset_name: Name of the dataset to load evaluation data for
            
        Returns:
            DataFrame containing evaluation data
            
        Raises:
            NotImplementedError: If dataset or evaluation config is not supported
        """

        if not cls.is_supported_dataset(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} is not supported")

        if "experiments_" in dataset_name:
            dataset_type, group = dataset_name.split('_')[1:]
            df = datasets.load_dataset("Bravansky/dataset-featurization", dataset_type, "train").to_pandas()
            df = df[df["group"] == int(group)]
            df["string"] = df["text"]
            return df.reset_index(drop=True)

        dataset_config = SUPPORTED_DATASETS[dataset_name]
        
        if not dataset_config.get("evaluation_source"):
            raise NotImplementedError(f"No evaluation configuration for dataset {dataset_name}")
            
        df = DataLoader.load_data(
            dataset_config["evaluation_source"],
            dataset_config.get("evaluation_config", None)
        ).to_pandas()
        
        return df.reset_index(drop=True)