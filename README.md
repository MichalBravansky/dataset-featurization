# Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction

This repository provides a pipeline for generating and selecting important features from a set of strings. The process helps reduce perplexity by identifying key patterns and elements within the strings. The pipeline operates in two stages: **feature generation** and **featurization**.

## Overview

1. **Feature Generation**: Proposes a set of possible features for the given strings and creates a table that indicates whether each feature is present in each string.
   
2. **Featurization**: Selects the most important features by iteratively adding them, minimizing perplexity, and producing a final set of features.

## Getting Started

### Prerequisites

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_key

# Wandb is optional
WANDB_PROJECT=your_project_name
WANDB_USERNAME=your_username
WANDB_MODE=your_mode
```

3. Set up the config.py (primarily the dataset and the prompt template used during featurization)
   
### Usage

1. Generate features from your dataset:
```bash
python generation.py
```

2. Featurize your dataset:
```bash
python featurization.py
```