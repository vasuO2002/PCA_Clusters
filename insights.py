from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()


df = pd.read_csv('rer.csv')


date_columns = ['requested_delivery_date', 'order_date', 'actual_delivery_date', 'promised_delivery_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])


class RequestPayload(BaseModel):
    display_desc: str
    start_date: str
    end_date: str
    frequency: str

def calculate_metrics(df, display_desc, start_date, end_date, frequency):
    
    filtered_df = df[(df['display_desc'] == display_desc) & (df['order_date'] >= start_date) & (df['order_date'] <= end_date)]
    
   
    if frequency == 'daily':
        resampled_df = filtered_df.resample('D', on='order_date')
    elif frequency == 'weekly':
        resampled_df = filtered_df.resample('W-Mon', on='order_date')
    elif frequency == 'monthly':
        resampled_df = filtered_df.resample('M', on='order_date')
    elif frequency == 'quarterly':
        resampled_df = filtered_df.resample('Q', on='order_date')
    else:
        raise HTTPException(status_code=400, detail="Invalid frequency. Supported frequencies are 'daily', 'weekly', 'monthly', and 'quarterly'.")
    
   
    new_products_per_period = resampled_df['product_id'].nunique().diff().fillna(0)
    
   
    new_products_per_period[new_products_per_period < 0] = 0
    
    
    active_products_per_period = resampled_df['product_id'].nunique()
    
    
    existing_products_per_period = active_products_per_period - new_products_per_period
    
    
    retention_rate_per_period = active_products_per_period / active_products_per_period.shift(1)
    
    
    result_df = pd.DataFrame({
        'order_date': new_products_per_period.index,
        'active_products': active_products_per_period.values,
        'new_products': new_products_per_period.values,
        'existing_products': existing_products_per_period.values,
        'retention_rate': retention_rate_per_period.values
    })
    
    
    result_df.replace([np.nan, np.inf, -np.inf], None, inplace=True)
    
    return result_df


@app.post("/cohort-analysis/")
async def cohort_analysis(payload: RequestPayload):
    try:
        start_date = pd.to_datetime(payload.start_date)
        end_date = pd.to_datetime(payload.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use 'yyyy-mm-dd' format.")
    
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date cannot be after end date.")
    
    result_df = calculate_metrics(df, payload.display_desc, start_date, end_date, payload.frequency)
    
    
    json_payload = {
        "order_date": result_df['order_date'].dt.strftime('%Y-%m-%d').tolist(),
        "active_products": result_df['active_products'].tolist(),
        "new_products": result_df['new_products'].tolist(),
        "existing_products": result_df['existing_products'].tolist(),
        "retention_rate": result_df['retention_rate'].tolist()
    }
    
    return {"dataframe": result_df.to_dict(orient='records'), "json_payload": json_payload}
