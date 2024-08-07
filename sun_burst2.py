import pandas as pd
import plotly.graph_objects as go


csv_file_path = 'example.csv'
df = pd.read_csv(csv_file_path)




if 'label' in df.columns and 'parent' in df.columns and 'value' in df.columns:
    labels = df['label'].tolist()
    parents = df['parent'].tolist()
    values = df['value'].tolist()

   
    total_value = sum(values)
    percentages = [value / total_value * 100 for value in values]

  
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=percentages,
        textinfo='label+percent entry'
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    fig.show()
else:
    print("CSV file must contain 'label', 'parent', and 'value' columns.")
