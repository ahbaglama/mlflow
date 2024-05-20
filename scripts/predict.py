import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5002")

logged_model = 'runs:/e9e5c076127649ed8239119c6aa13183/model_random_forest'

# Load model
loaded_model = mlflow.pyfunc.load_model(logged_model)

df = pd.read_csv('../data/mlops_data.csv')

data = df.drop('target', axis=1)
target = df['target']

# Standardize the features using the same scaler as used during training
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Make predictions
predictions = loaded_model.predict(data_scaled)

accuracy = accuracy_score(target, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)
