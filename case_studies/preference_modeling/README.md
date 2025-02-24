# Compositional Preference Modeling Case Study

This repository contains the code and resources for the compositional preference modeling case study in **Dataset Featurization**. The dataset is hosted on [Hugging Face datasets](https://huggingface.co/datasets/Bravansky/compositional-preference-modeling).

## Overview

The study explores how unsupervised feature extraction can match or exceed the performance of expert-designed features in preference modeling. We evaluate on two datasets following the [Compositional Preference Modeling (CPM)](https://github.com/dongyoung-go/cpm) methodology:

- [HH-RLHF](https://arxiv.org/abs/2204.05862): Machine-generated responses ranked for alignment
- [SHP](https://proceedings.mlr.press/v162/ethayarajh22a.html): Human-written responses from natural interactions 

## Structure

The structure of this evaluation set is heavily taken from [CPM](https://github.com/dongyoung-go/cpm). We provide all generated features with their assigned attributes in `constants.py`. Additionally, the repository is split into two main directories:

### `annotation` Folder

- `download_datasets.py`: Downloads the CPM datasets from Hugging Face.
- `generate_attributes.py`: Generates attributes for produced features.
- `annotate.py`: Contains the code to assign features and attributes to the HH-RLHF and SHP datasets.

### `analysis` Folder

- `train_logistic_regression`: Trains linear regressions on the generated features.
- `models`: Contains all the trained liner regressions.
- `predict_reward.py`: Makes predictions on top of Best of N (BoN) responses with the trained models.
- `select_bon_candidate.py`: Selects the best candidate from the BoN responses.
- `alpaca_eval.py`: Evaluates the the win-rate of the models using the Alpaca Evaluation Framework and GPT-4o.

### 1. Define a `.env` File
Create a `.env` file in the root directory and add the following environment variables:
```
OPENAI_API_KEY=**

WANDB_PROJECT=**
WANDB_USERNAME=**
WANDB_MODE=**
```

### 2. Run the Evaluation

Follow the instructions in the `annotation` and `analysis` folders to reproduce the results.