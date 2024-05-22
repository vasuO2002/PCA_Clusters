import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict
from fastapi import Body
app2 = FastAPI()

data = pd.read_csv('sample1.csv')
specified_product_ids = [8314687, 8314685, 8314682, 8314679, 8250452, 8250451] #8250450, 7856896, 7858787, 8236790]


data = data[data['product_id'].isin(specified_product_ids)]
# data['product_id'] = data['product_id'].astype('int64')
result = data.groupby(['product_id', 'order_date']).sum(numeric_only=True).reset_index()
transformed_df = result.pivot(index='product_id', columns='order_date', values='final_quantity_requested')
transformed_df.reset_index(inplace=True)
transformed_df.index.name = None
transformed_df = transformed_df.apply(lambda x: x.fillna(x.median()), axis=0)
X = transformed_df.drop(columns=['product_id']).values

labels = None
X_pca = None
num_clusters = None
distance_metric = None
from pydantic import BaseModel
@app2.get("/cluster_product_ids/", response_class=HTMLResponse)
async def cluster_product_ids(
    num_clusters_: int = Query(..., description="Number of clusters to use."),
    pca_components: int = Query(..., description="Number of PCA components."),
    distance_metric_: str = Query(..., description="Distance metric to use (cosine, euclidean, or manhattan).")
):
    """
    Cluster product IDs based on given parameters.
    """
    global labels, X_pca, num_clusters, distance_metric
    
    num_clusters = num_clusters_
    distance_metric = distance_metric_

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
            text=transformed_df.loc[labels == label, 'product_id'],
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


from fastapi import Body, FastAPI
from typing import List, Dict

app = FastAPI()

class ClusterAssignment(BaseModel):
    product_id: int
    new_cluster: int

@app2.post("/edit_cluster_assignment/", response_class=HTMLResponse)
async def edit_cluster_assignment(
    clusters: List[ClusterAssignment] = Body(..., example=[
        {"product_id": 7982571, "new_cluster": 1},
        {"product_id": 8079282, "new_cluster": 0}
    ])
):
    """
    Edit cluster assignments for specific product IDs.
    """
    global labels
    
    for entry in clusters:
        product_id = entry.product_id
        new_cluster = entry.new_cluster
        
        
        idx = transformed_df.index[transformed_df['product_id'] == product_id].tolist()
        if len(idx) == 0:
            return f"Product ID {product_id} not found."
        idx = idx[0]

        
        labels[idx] = new_cluster

    
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
            text=transformed_df.loc[labels == label, 'product_id'],
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


# from fastapi import Body

# @app2.post("/remove_clusters_and_product_ids/", response_class=HTMLResponse)
# async def remove_clusters_and_product_ids(
#     clusters: List[int] = Body(..., description="List of cluster numbers to remove."),
#     product_ids: List[int] = Body(..., description="List of product IDs to remove.")
# ):
#     """
#     Remove specific clusters or product IDs from the graph.
#     """
#     global labels

#     # Update the visualization with the modified clusters
#     fig = go.Figure()
#     for label in range(num_clusters):
#         if label not in clusters:
#             fig.add_trace(go.Scatter3d(
#                 x=X_pca[labels == label, 0],
#                 y=X_pca[labels == label, 1],
#                 z=X_pca[labels == label, 2],
#                 mode='markers',
#                 marker=dict(
#                     color=label,
#                     colorscale='viridis',
#                     size=5,
#                     opacity=0.8
#                 ),
#                 text=transformed_df.loc[labels == label, 'product_id'],
#                 name=f'Cluster {label}'
#             ))

#     for product_id in product_ids:
#         idx = transformed_df.index[transformed_df['product_id'] == product_id].tolist()
#         if len(idx) > 0:
#             idx = idx[0]
#             labels = np.delete(labels, idx)

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Principal Component 1',
#             yaxis_title='Principal Component 2',
#             zaxis_title='Principal Component 3',
#         ),
#         title=f'Clustered Product IDs in 3D (Clusters={num_clusters}, Distance={distance_metric.capitalize()})'
#     )

#     plot_html = fig.to_html(full_html=False)
#     return f"<h1>Cluster Analysis</h1>{plot_html}"


# from fastapi import Body

# @app2.post("/remove_clusters_and_product_ids/", response_class=HTMLResponse)
# async def remove_clusters_and_product_ids(
#     clusters: List[int] = Body(..., description="List of cluster numbers to remove."),
#     product_ids: List[int] = Body(..., description="List of product IDs to remove.")
# ):
#     """
#     Remove specific clusters or product IDs from the graph.
#     """
#     global labels

#     # Update the visualization with the modified clusters
#     fig = go.Figure()
#     for label in range(num_clusters):
#         cluster_product_ids = transformed_df.loc[labels == label, 'product_id']
#         for product_id in cluster_product_ids:
#             if product_id not in product_ids and label not in clusters:
#                 fig.add_trace(go.Scatter3d(
#                     x=X_pca[labels == label, 0],
#                     y=X_pca[labels == label, 1],
#                     z=X_pca[labels == label, 2],
#                     mode='markers',
#                     marker=dict(
#                         color=label,
#                         colorscale='viridis',
#                         size=5,
#                         opacity=0.8
#                     ),
#                     text=transformed_df.loc[labels == label, 'product_id'],
#                     name=f'Cluster {label}'
#                 ))

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Principal Component 1',
#             yaxis_title='Principal Component 2',
#             zaxis_title='Principal Component 3',
#         ),
#         title=f'Clustered Product IDs in 3D (Clusters={num_clusters}, Distance={distance_metric.capitalize()})'
#     )

#     plot_html = fig.to_html(full_html=False)
#     return f"<h1>Cluster Analysis</h1>{plot_html}"


# from fastapi import Body
# import numpy as np
# @app2.post("/remove_clusters_and_product_ids/", response_class=HTMLResponse)
# async def remove_clusters_and_product_ids(
#     clusters: List[int] = Body(..., description="List of cluster numbers to remove."),
#     product_ids: List[int] = Body(..., description="List of product IDs to remove.")
# ):
#     """
#     Remove specific clusters or product IDs from the graph.
#     """
#     global labels

#     # Update the visualization with the modified clusters
#     fig = go.Figure()
#     for product_id in product_ids:
#         idx = transformed_df.index[transformed_df['product_id'] == product_id].tolist()
#         if len(idx) > 0:
#             labels = np.delete(labels, idx[0])
#     for label in range(num_clusters):
#         if label not in clusters:
#             # Plot points for clusters not in the specified list
#             fig.add_trace(go.Scatter3d(
#                 x=X_pca[labels == label, 0],
#                 y=X_pca[labels == label, 1],
#                 z=X_pca[labels == label, 2],
#                 mode='markers',
#                 marker=dict(
#                     color=label,
#                     colorscale='viridis',
#                     size=5,
#                     opacity=0.8
#                 ),
#                 text=transformed_df.loc[labels == label, 'product_id'],
#                 name=f'Cluster {label}'
#             ))
    
#     # Remove product IDs
    

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Principal Component 1',
#             yaxis_title='Principal Component 2',
#             zaxis_title='Principal Component 3',
#         ),
#         title=f'Clustered Product IDs in 3D (Clusters={num_clusters}, Distance={distance_metric.capitalize()})'
#     )

#     plot_html = fig.to_html(full_html=False)
#     return f"<h1>Cluster Analysis</h1>{plot_html}"
from fastapi import Body

@app2.post("/remove_product_ids/", response_class=HTMLResponse)
async def remove_product_ids(
    product_ids: List[int] = Body(..., description="List of product IDs to remove.")
):
    """
    Remove specific product IDs from the graph and recalculate clusters.
    """
    global transformed_df, X, labels

    
    transformed_df = transformed_df[~transformed_df['product_id'].isin(product_ids)]
    X = transformed_df.drop(columns=['product_id']).values
    
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(X_pca)
    

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
            text=transformed_df.loc[labels == label, 'product_id'],
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


@app2.post("/remove_clusters/", response_class=HTMLResponse)
async def remove_clusters(
    clusters: List[int] = Body(..., description="List of cluster numbers to remove.")
):
    """
    Remove specific clusters from the graph.
    """
    global labels

    fig = go.Figure()
    for label in range(num_clusters):
        if label not in clusters:
            
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
                text=transformed_df.loc[labels == label, 'product_id'],
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
