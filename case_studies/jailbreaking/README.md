# Compact Jailbreaks Case Study

This repository contains the code and resources for the jailbreak case study in **Dataset Featurization**. The dataset is hosted on [Hugging Face datasets](https://huggingface.co/datasets/Bravansky/compact-jailbreaks).

## Overview
This study focuses on extracting key features from jailbreak datasets, using them to generate attacks, and evaluating their effectiveness across various models. The extracted features are provided in `constants.py`, and the repository includes code to generate and assess attacks.

## Setup Instructions
To get started, ensure you have the necessary API keys and environment variables configured.

### 1. Define a `.env` File
Create a `.env` file in the root directory and add the following environment variables:
```
TOGETHER_API_KEY=**
OPENROUTER_API_KEY=**
OPENAI_API_KEY=**

WANDB_PROJECT=**
WANDB_USERNAME=**
WANDB_MODE=**
```

### 2. Run Attacks
The repository provides two main scripts for running attacks:

1. `jailbreak_generation.py`: Generates jailbreak attacks using feature-based prompting
2. `jailbreak_attack.py`: Evaluates generated attacks against various models

### 3. Utility Modules
The `utils` directory contains several helper modules:
- `data_utils.py`: Functions for data processing and prompt generation
- `together_generator.py`: Client for Together API interactions
- `open_router_generator.py`: Client for OpenRouter API interactions
- `openai_generator.py`: Client for OpenAI API interactions

### 4. Evaluate Attacks
To evaluate the effectiveness and diversity of the generated attacks, please use the WildTeaming repository:
[WildTeaming Evaluation Framework](https://github.com/allenai/wildteaming)