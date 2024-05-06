from fastapi import FastAPI, Query, HTTPException, Response
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

def detect_changes(time_series, min_size):
    
    signal = time_series.values

    
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)
    result = algo.predict(pen=2)

    
    change_points = [i for i in result if i < len(signal)]

    
    return change_points

@app.post("/detect_changes/")
async def detect_changes_api(min_size: int = Query(..., description="Minimum size parameter")):
    
    df = pd.read_csv("filtered_data.csv")

    df['order_date'] = pd.to_datetime(df['order_date'])

    
    weekly_data = df.resample('W-Mon', on='order_date').sum()

    
    change_points = detect_changes(weekly_data['final_quantity_requested'], min_size)

    
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data.index, weekly_data['final_quantity_requested'], label='Final Quantity Requested (Weekly)')
    plt.xlabel('Order Date (Weekly)')
    plt.ylabel('Final Quantity Requested (Weekly)')
    plt.title(f'Weekly Time Series Data with Changepoints (Min Size: {min_size})')

    
    for cp in change_points:
        plt.axvline(x=weekly_data.index[cp], color='r', linestyle='--', linewidth=2)

    plt.tight_layout()

    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

   
    plt.close()

  
    return Response(content=img_buffer.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
