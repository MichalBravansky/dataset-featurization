from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Dict

class ModelReconstructionEvaluator:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def evaluate(self, df: pd.DataFrame, features: list, top_n: int) -> Dict[str, float]:
        
        categories = df['category'].apply(lambda x: x.split('/')[-1])
        y = self.label_encoder.fit_transform(categories)
        X = df[features[:top_n]]
        
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        } 