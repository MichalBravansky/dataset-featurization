from typing import Dict, List
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def generate_min_max_attributes(text: str) -> Dict[str, str]:
    """Generate min and max attributes for a given text using GPT-4"""
    prompt = f"""Given the feature: "{text}"

    Generate minimum and maximum attributes that can be used to evaluate LLM response quality through a rating scale utilizing the given feature.
    
    Return only a JSON object in this format:
    {{
        "attr_min": "<opposite/minimum state>",
        "attr_max": "<maximum/extreme state>"
    }}

    Example:
    Feature: "ends suddently, creating confusion"
    {{
        "attr_min": "ends smoothly and conclusively",
        "attr_max": "ends very suddently"
    }}"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates attribute descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        text = completion.choices[0].message.content.strip("```").strip("json")
        return json.loads(text)
    except Exception as e:
        print(f"Error processing '{text}': {e}")
        return {
            'attr_min': "ERROR",
            'attr_max': "ERROR"
        }

def process_dataset(texts: List[str], output_prefix: str):
    """Process a list of texts and save results to files"""
    results = []
    
    for text in tqdm(texts, desc="Processing attributes"):
        min_max = generate_min_max_attributes(text)
        results.append({
            'attribute_desc': text,
            'attr_min': min_max['attr_min'],
            'attr_max': min_max['attr_max']
        })

    # Save results
    with open(f"{output_prefix}_attributes.txt", 'w') as f:
        json.dump(results, f, indent=2)