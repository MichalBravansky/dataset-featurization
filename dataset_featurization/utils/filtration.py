from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import CLUSTER_SIZE, client
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import time
import joblib
import os

class Filter:

    def __init__(self, model = "text-embedding-3-small"):
        self._model = model

    def _get_embeddings(self, texts: list, batch_size = 100):
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = [row.embedding for row in client.embeddings.create(input=batch_texts, model=self._model).data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _find_most_central_row(self, cluster_df):
        embeddings = np.stack(cluster_df["embedding"].to_list())
        dist_matrix = cosine_distances(embeddings)
        sum_distances = dist_matrix.sum(axis=1)
        most_central_idx = sum_distances.argmin()
        return cluster_df.iloc[most_central_idx]

    def filter(self, descriptions, save_model = False):
        descriptions = [description for description in descriptions if "The selected string" in description]

        df = pd.DataFrame(descriptions, columns=["description"])

        df["embedding"] = self._get_embeddings(df["description"].to_list())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df["embedding"].to_list())  

        NUM_CLUSTERS = len(df.index) // CLUSTER_SIZE

        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        df["cluster"] = labels

        selected_descriptions = []

        for cluster_id in range(NUM_CLUSTERS):
            sub_df = df[df["cluster"] == cluster_id]
            most_central_row = self._find_most_central_row(sub_df)
            selected_descriptions.append(most_central_row["description"])
            
        if save_model:
            timestamp = int(time.time() * 1000)
            os.makedirs("data", exist_ok=True)
            labels_path = f'data/cluster_labels_{timestamp}.txt'
            np.savetxt(labels_path, labels, fmt='%d')

        return selected_descriptions
    
