import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from typing import List, Dict

app2 = FastAPI()


data = pd.read_csv('sample1.csv')


result = data.groupby(['product_id', 'order_date']).sum(numeric_only=True).reset_index()
transformed_df = result.pivot(index='product_id', columns='order_date', values='final_quantity_requested')
transformed_df.reset_index(inplace=True)
transformed_df.index.name = None
transformed_df = transformed_df.apply(lambda x: x.fillna(x.median()), axis=0)
X = transformed_df.drop(columns=['product_id']).values

@app2.get("/cluster_product_ids/", response_class=HTMLResponse)
async def cluster_product_ids(
    num_clusters: int = Query(..., description="Number of clusters to use."),
    pca_components: int = Query(..., description="Number of PCA components."),
    distance_metric: str = Query(..., description="Distance metric to use (cosine, euclidean, or manhattan).")
):
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
    
    
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(distances)
    
    
    fig = go.Figure()
    for label in range(num_clusters):
        fig.add_trace(go.Scatter3d(
            x=X_pca[labels == label, 0],
            y=X_pca[labels == label, 1],
            z=X_pca[labels == label, 2],
            mode='markers',
            marker=dict(
                color=label,
                colorscale='viridis',
                size=5,
                opacity=0.8
            ),
            name=f'Cluster {label}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3',
        ),
        title=f'Clustered Product IDs in 3D (Clusters={num_clusters}, Distance={distance_metric.capitalize()})'
    )

    
    plot_html = fig.to_html(full_html=False)
    return f"<h1>Cluster Analysis</h1>{plot_html}"

