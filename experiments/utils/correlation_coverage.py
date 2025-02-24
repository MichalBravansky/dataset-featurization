import pandas as pd
import numpy as np
from typing import List

class CorrelationCoverageEvaluator:
    def evaluate(self, df: pd.DataFrame, features: List[str], categories: List[str], top_n: int) -> float:
        correlations = []
        for feature in features[:top_n]:
            feature_correlations = []
            for category in categories:
                category_mask = (df['category'] == category).astype(int)
                if df[feature].std() == 0:
                    feature_correlations.append(0.0)
                    continue
                correlation = df[feature].corr(category_mask)
                feature_correlations.append(abs(correlation) if not pd.isna(correlation) else 0.0)
            correlations.append(feature_correlations)
        
        max_correlations = np.max(correlations, axis=0) if correlations else np.zeros(len(categories))
        return np.mean(max_correlations) 