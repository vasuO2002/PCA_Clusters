import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


final_df = pd.read_csv('final_data_with_contribution1.csv')
final_df['order_date'] = pd.to_datetime(final_df['order_date'])
final_df.fillna(0, inplace=True)

final_df['day_of_week'] = final_df['order_date'].dt.dayofweek
final_df['month'] = final_df['order_date'].dt.month
final_df['is_weekend'] = final_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


X = final_df.drop(columns=['contribution', 'order_date'])  # Exclude datetime column
y = final_df['contribution']

categorical_cols = ['display_desc']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)


r2 = r2_score(y_test, y_pred)
print("R2 Score on Test Set:", r2)


joblib.dump(rf_regressor, 'random_forest_regressor_model.pkl')
