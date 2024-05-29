import pandas as pd


df = pd.read_csv('xyz1.csv')


category_totals = df.groupby(['order_date', 'display_desc'])['final_price_usd'].sum().reset_index()


merged_df = df.merge(category_totals, on=['order_date', 'display_desc'], suffixes=('', '_total'))


merged_df['contribution'] = merged_df['final_price_usd'] / merged_df['final_price_usd_total']


final_df = merged_df[['product_id', 'order_date', 'display_desc', 'final_price_usd_total', 'contribution']]


final_df.to_csv('final_data_with_contribution1.csv', index=False)
