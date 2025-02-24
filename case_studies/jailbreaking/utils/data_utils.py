import pandas as pd
import numpy as np
from typing import List, Tuple
import random
from constants import GENERATION_PROMPT
    
def generate_jailbreak(
    df: pd.DataFrame,
    selected_features: List[str],
    test_prompts: List[str],
    use_features: bool = True
) -> List[str]:
    """
    Generate jailbreak prompts for test inputs using few-shot learning
    
    Args:
        df: DataFrame with training data
        selected_features: List of feature names to use
        test_prompts: List of prompts to transform
        use_features: Whether to include features in examples
    
    Returns:
        List of generated prompts
    """
    generated_prompts = []
    
    formatted_features = []
    for feature in selected_features:
        feature = "- " + feature.replace("The selected string", "The revised adversarial prompt")
        formatted_features.append(feature)
    
    for test_prompt in test_prompts:
        if use_features:
            
            sampled_features = random.sample(formatted_features, min(4, len(formatted_features)))
            feature_text = "\n".join(sampled_features)
        else:
            feature_text = ""

        prompt = GENERATION_PROMPT.format(test_prompt=test_prompt["vanilla"], feature_text=feature_text)
        
        generated_prompts.append(prompt)
    
    return generated_prompts

def create_test_prompt_dict(prompt: str, features: List[str], feature_vector: List[bool]) -> dict:
    """
    Create a dictionary containing the test prompt and its features
    
    Args:
        prompt: The original prompt string
        features: List of feature names
        feature_vector: List of boolean values indicating feature presence
        
    Returns:
        Dictionary with prompt and features
    """
    result = {
        "vanilla": prompt,
        "feature_vector": feature_vector
    }

    result.update(dict(zip(features, feature_vector)))
    
    return result