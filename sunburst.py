from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import plotly.express as px

app = FastAPI()


df = pd.read_csv('Product.csv')

@app.get("/sunburst/")
async def get_sunburst(parent: str = Query(None, description="Parent to display")):
    
    if parent:
        filtered_df = df[df['display_desc'] == parent]
    else:
        filtered_df = df.copy()  

    
    agg_data = filtered_df.groupby(['display_desc', 'brand_name', 'id']).size().reset_index(name='count')
    agg_data = agg_data.sort_values(by=['display_desc', 'brand_name'])

    fig = px.sunburst(
        agg_data,
        path=['display_desc', 'brand_name', 'id'],
        values='count',
    )

    
    html_content = fig.to_html(full_html=False)
    return HTMLResponse(content=html_content)

