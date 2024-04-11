from fastapi import FastAPI, Query
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import List, Dict

app1 = FastAPI()


data = pd.read_csv('sample1.csv')


result = data.groupby(['product_id', 'order_date']).sum(numeric_only=True).reset_index()
transformed_df = result.pivot(index='product_id', columns='order_date', values='final_quantity_requested')
transformed_df.reset_index(inplace=True)
transformed_df.index.name = None
transformed_df = transformed_df.apply(lambda x: x.fillna(x.median()), axis=0)
X = transformed_df.drop(columns=['product_id']).values

@app1.get("/cluster_product_ids/")
async def cluster_product_ids(
    num_clusters: int = Query(..., description="Number of clusters to use."),
    pca_components: int = Query(..., description="Number of PCA components."),
    distance_metric: str = Query(..., description="Distance metric to use (cosine, euclidean, or manhattan).")
) -> Dict[int, List[int]]:
    """
    Cluster product IDs based on given parameters.
    """
    
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)
    
    
    if distance_metric == 'cosine':
        distances = pairwise_distances(X_pca, metric='cosine')
    elif distance_metric == 'euclidean':
        distances = pairwise_distances(X_pca, metric='euclidean')
    elif distance_metric == 'manhattan':
        distances = pairwise_distances(X_pca, metric='manhattan')
    
    
    kmeans = KMeans(n_clusters=num_clusters,n_init=10)
    labels = kmeans.fit_predict(distances)
    
    
    clusters = {}
    for product_id, cluster_label in zip(transformed_df['product_id'], labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = [product_id]
        else:
            clusters[cluster_label].append(product_id)
    
    return clusters
