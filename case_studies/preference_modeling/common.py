import re
from openai import OpenAI
from constants import SHP_TEMPLATE, HH_TEMPLATE
from dotenv import load_dotenv

load_dotenv()

def format_attributes(features):
    formatted = []
    for i, (_, kwargs) in enumerate(features.items(), 1):
        formatted.append(
            f"{i}. Score whether the reply {kwargs['attribute_desc']} "
            f"(1 = {kwargs['attr_min']}, 10 = {kwargs['attr_max']})"
        )
    return "\n".join(formatted)

def get_lm(model_name, model_type):
    if model_type == 'openai':
        return OpenAI()
    elif model_type == 'huggingface':
        # You might want to handle this differently or remove if not needed
        raise NotImplementedError("Huggingface implementation needs to be updated")

def feature_score(client, features, data_type, **feature_kwargs):
    try:
        all_scores = []
        feature_items = list(features.items())
        
        # Process features in batches of 5
        for i in range(0, len(feature_items), 5):
            batch_features = dict(feature_items[i:i+5])
            attributes_text = format_attributes(batch_features)
            
            template = SHP_TEMPLATE if data_type == 'shp' else HH_TEMPLATE
            prompt = template.format(
                **feature_kwargs,
                attributes=attributes_text,
                num_attributes=len(batch_features)
            )
            
            # Try up to 5 times to get valid scores
            for attempt in range(5):
                response = client.chat.completions.create(
                    model=feature_kwargs.get('model_name', 'gpt-4o-2024-08-06'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                lm_response = response.choices[0].message.content
                batch_scores = [int(x.strip()) for x in lm_response.strip().split('\n') if x.strip().isdigit()]
                
                if len(batch_scores) == len(batch_features):
                    all_scores.extend(batch_scores)
                    break
                
                if attempt == 4:  # If all attempts failed
                    all_scores.extend([None] * len(batch_features))
        
        return all_scores
        
    except Exception as e:
        print(e)
        return [None] * len(features)
    
def get_feature_extractor(lm, data_type):
    """
    Returns a function that can extract features using the specified model.
    
    Args:
        lm: OpenAI client instance
        data_type: Type of data ('shp' or 'hh')
    
    Returns:
        Function that takes kwargs and returns feature scores
    """
    def extract_features(kwargs):
        return feature_score(
            client=lm,
            data_type=data_type,
            **kwargs
        )
    
    return extract_features