import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

# Load processed data
df = pd.read_csv('data/processed_data.csv')

# Load preprocessor and get features
prep = DataPreprocessor()
prep.load_encoders('models/label_encoders.pkl')
feature_cols = prep.get_feature_columns()

X = df[feature_cols]
y = df['salary_usd']

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
with open('models/salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MAE: ${train_mae:,.2f}")
print(f"Test MAE: ${test_mae:,.2f}")
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")
print(f"\nDataset: {len(df)} records")
print(f"Features used: {len(feature_cols)}")
