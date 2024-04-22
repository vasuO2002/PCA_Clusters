from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.metrics import r2_score
from pycaret.regression import setup, compare_models, finalize_model, predict_model
from enum import Enum
from typing import Optional, List

app = FastAPI()


final_df = pd.read_csv('final_data_with_contribution1.csv')
final_df['order_date'] = pd.to_datetime(final_df['order_date'])
final_df.fillna(0, inplace=True)

class Periodicity(str, Enum):
    daily = "D"  
    weekly = "W"
    monthly = "M"

@app.get("/predict_contribution/")
async def predict_contribution(
    display_desc: str,
    product_id: str,
    start_date: str,
    end_date: str,
    periodicity: Periodicity,
    data_completeness: Optional[float] = None,
) -> dict:
    
    
    
    
    test_data = pd.DataFrame({
        'product_id': [product_id] * len(pd.date_range(start=start_date, end=end_date, freq=periodicity.value)),
        'order_date': pd.date_range(start=start_date, end=end_date, freq=periodicity.value),
        
        'display_desc': [display_desc] * len(pd.date_range(start=start_date, end=end_date, freq=periodicity.value))
    })

    
    additional_features = pd.DataFrame({
        
        'day_of_week': test_data['order_date'].dt.dayofweek,
        'month': test_data['order_date'].dt.month,
        'is_weekend': test_data['order_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    })

    
    test_data = pd.concat([test_data.reset_index(drop=True), additional_features], axis=1)

   
    train_data = final_df.drop(columns=['final_price_usd_total'])
    train_data = pd.concat([train_data.reset_index(drop=True), additional_features], axis=1)

    
    regression_setup = setup(data=train_data, target='contribution', train_size=0.8)

    
    best_model = compare_models()

   
    final_model = finalize_model(best_model)

    
    predictions = predict_model(final_model, data=test_data)

    
    validation_data = predict_model(final_model)
    residuals = validation_data['contribution'] - validation_data['prediction_label']
    ss_res = (residuals ** 2).sum()
    ss_tot = ((validation_data['contribution'] - validation_data['contribution'].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    
    response_dict = {
        'product_id': list(predictions['product_id']),
        'display_desc': list(predictions['display_desc']),
        'contribution': list(predictions['prediction_label']), 
        'date': list(predictions['order_date'].dt.strftime('%Y-%m-%d')),  
        'periodicity': str(periodicity),  
        'r2_score': str(r2) 
    }

    return response_dict
