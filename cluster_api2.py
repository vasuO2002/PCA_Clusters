from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from typing import Dict, List

app3 = FastAPI()

data = pd.read_csv('sample1.csv')


data['order_date'] = pd.to_datetime(data['order_date'])

@app3.get("/cluster_product_ids/")
async def cluster_product_ids(
    start_date: str = Query(None, description="Start date for the order_date column (YYYY-MM-DD)."),
    end_date: str = Query(None, description="End date for the order_date column (YYYY-MM-DD).")
) -> Dict[str, List]:
    """
    Find the optimal clustering of product IDs based on silhouette scores.
    """
    if not all([start_date, end_date]):
        return JSONResponse(content={"error": "You must provide values for start_date and end_date."}, status_code=400)
    
    
    selected_data = data[(data['order_date'] >= start_date) & (data['order_date'] <= end_date)]

    
    result = selected_data.groupby(['product_id', 'order_date']).sum(numeric_only=True).reset_index()
    transformed_df = result.pivot(index='product_id', columns='order_date', values='final_quantity_requested')
    transformed_df.reset_index(inplace=True)
    transformed_df.index.name = None
    transformed_df = transformed_df.apply(lambda x: x.fillna(x.median()), axis=0)
    X = transformed_df.drop(columns=['product_id']).values

    num_clusters = [3, 5, 7, 9, 12, 15, 17, 19]
    pca_components = [3, 7]
    distance_measures = ['cosine', 'euclidean', 'manhattan']

    silhouette_scores = []
    best_params = {}

    for num_cluster in num_clusters:
        for pca_component in pca_components:
            # Perform PCA on the selected range
            pca = PCA(n_components=pca_component)
            X_pca = pca.fit_transform(X)

            for distance_measure in distance_measures:
            
                if distance_measure == 'cosine':
                    distances = pairwise_distances(X_pca, metric='cosine')
                elif distance_measure == 'euclidean':
                    distances = pairwise_distances(X_pca, metric='euclidean')
                elif distance_measure == 'manhattan':
                    distances = pairwise_distances(X_pca, metric='manhattan')

                
                kmeans = KMeans(n_clusters=num_cluster)
                labels = kmeans.fit_predict(distances)

                # Calculate silhouette score
                silhouette = silhouette_score(X_pca, labels)

                # Store the best parameters based on silhouette score
                if not best_params or silhouette > best_params['silhouette']:
                    best_params = {
                        'num_cluster': num_cluster,
                        'pca_component': pca_component,
                        'distance_measure': distance_measure,
                        'silhouette': silhouette,
                        'labels': labels
                    }

    # Use the best parameters
    num_cluster = best_params['num_cluster']
    pca_component = best_params['pca_component']
    distance_measure = best_params['distance_measure']
    silhouette = best_params['silhouette']
    labels = best_params['labels']

    # Prepare result dictionary with product IDs and their respective cluster numbers
    clusters = {}
    for product_id, cluster_label in zip(transformed_df['product_id'], labels):
        cluster_label_str = str(cluster_label)
        if cluster_label_str not in clusters:
            clusters[cluster_label_str] = [product_id]
        else:
            clusters[cluster_label_str].append(product_id)

    # Convert dictionary keys and values to JSON serializable types
    clusters_serializable = {}
    for key, value in clusters.items():
        clusters_serializable[key] = [str(item) for item in value]

    # Return the best parameters and clusters in a single response body
    response_content = {
        "message": f"Best Clustering Case: Num Clusters={num_cluster}, PCA Components={pca_component}, Distance Measure={distance_measure}, Silhouette Score={silhouette}",
        "clusters": clusters_serializable
    }

    return JSONResponse(content=response_content)
