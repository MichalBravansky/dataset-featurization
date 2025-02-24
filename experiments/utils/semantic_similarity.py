from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from dotenv import load_dotenv
import os

class SemanticSimilarityEvaluator:
    def __init__(self):
        load_dotenv()
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.prompt = """Do these two classes share the same meaning? Output only 'yes' or 'no.'
Class 1: {text1}
Class 2: {text2}"""

    def get_similarities(self, df, features: List[str], categories: List[str]) -> List[List[float]]:
        results_map = {feature: {category: None for category in categories} for feature in features}
        pairs = [(feature, category) for feature in features for category in categories]
        
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = {
                executor.submit(self._get_similarity_for_pair, feature, category): (feature, category)
                for feature, category in pairs
            }
            
            for future in as_completed(futures):
                feature, category, similarity = future.result()
                results_map[feature][category] = similarity
        
        return [[results_map[feature][category] for category in categories] for feature in features]

    def _get_similarity_for_pair(self, feature: str, category: str) -> Tuple[str, str, float]:
        try:
            response = self.openai_client.chat.completions.create(
                extra_headers={"X-Title": "Semantic Similarity"},
                model="anthropic/claude-3.5-haiku",
                messages=[{
                    "role": "user",
                    "content": self.prompt.format(text1=feature, text2=category)
                }],
                temperature=0,
                max_tokens=1
            )
            result = response.choices[0].message.content.lower().strip()
            similarity = {'yes': 1.0, 'partially': 0, 'no': 0.0}.get(result, 0.0)
        except Exception as e:
            print(f"Error comparing {feature} and {category}: {e}")
            similarity = 0.0
        return (feature, category, similarity) 