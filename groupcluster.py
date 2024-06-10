import pandas as pd


df = pd.read_csv('ALL_SKU_DATA.csv')

columns_to_use = ['product_name', 'color','description','bullet_points']


df['combined_text'] = df[columns_to_use].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


corpus = df['combined_text'].tolist()

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import defaultdict


dictionary = defaultdict(int)
tagged_data = []

for i, doc in enumerate(corpus):
    tokens = gensim.utils.simple_preprocess(doc)
    tagged_data.append(TaggedDocument(words=tokens, tags=[str(i)]))
    for token in tokens:
        dictionary[token] += 1


unique_words = list(dictionary.keys())


vector_size = len(unique_words)

model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=4, epochs=40)


model.build_vocab(tagged_data)


model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

# Load the trained Doc2Vec model
# model = Doc2Vec.load('path/to/your/doc2vec_model')


df = pd.read_csv('ALL_SKU_DATA.csv')


df['combined_text'] = df[['product_name', 'color', 'description', 'bullet_points']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


def infer_vector(text):
    tokens = simple_preprocess(text)
    return model.infer_vector(tokens)

df['vector'] = df['combined_text'].apply(infer_vector)

new_df = df[['sku_code', 'vector']]

new_df.to_csv('vector.csv', index=False)

import pandas as pd
import numpy as np
from ast import literal_eval


df = new_df

if isinstance(df['vector'][0], str):
    df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

vectors = np.stack(df['vector'].values)
df = pd.DataFrame(vectors, index=df['sku_code'])
print(df.shape)

from sklearn.decomposition import PCA


vectors = np.stack(df['vector'].values)


pca = PCA(n_components=50)


reduced_vectors = pca.fit_transform(vectors)
reduced_df = pd.DataFrame(reduced_vectors, index=df['sku_code'])


reduced_df.reset_index(inplace=True)



print(reduced_df.shape)

from sklearn.cluster import KMeans


k = 1500


vectors = reduced_df.drop(columns=['sku_code']).values


kmeans = KMeans(n_clusters=k, random_state=0)
reduced_df['cluster'] = kmeans.fit_predict(vectors)


print(reduced_df.head())

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
vectors = reduced_df.drop(columns=['sku_code', 'cluster']).values


pca_3d = PCA(n_components=3)
vectors_3d = pca_3d.fit_transform(vectors)


df_3d = pd.DataFrame(vectors_3d, columns=['PC1', 'PC2', 'PC3'])
df_3d['cluster'] = reduced_df['cluster']
df_3d['sku_code'] = reduced_df['sku_code']

fig = go.Figure()


for cluster in df_3d['cluster'].unique():
    cluster_data = df_3d[df_3d['cluster'] == cluster]
    fig.add_trace(
        go.Scatter3d(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            z=cluster_data['PC3'],
            mode='markers',
            name=f'Cluster {cluster}',
            text=cluster_data['sku_code'],
            hoverinfo='text',
            marker=dict(size=5)
        )
    )


fig.update_layout(
    title='3D Scatter Plot of Clusters',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ),
    legend_title_text='Cluster',
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=['visible', [True] * len(df_3d['cluster'].unique())],
                    label='Show All',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False] * len(df_3d['cluster'].unique())],
                    label='Hide All',
                    method='restyle'
                )
            ]),
            direction='down',
            showactive=True,
        )
    ]
)


fig.show()
fig.write_html('plot.html')

