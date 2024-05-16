import mlflow
# Set the MLflow URI if it's not the default one
mlflow.set_tracking_uri("http://127.0.0.1:5002")

logged_model = 'runs:/e9e5c076127649ed8239119c6aa13183/model_random_forest'


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Example prediction
import pandas as pd
# Assuming 'data' is already defined as a dictionary or another DataFrame creation method
df = pd.read_csv('../data/mlops_data.csv')

# Assuming 'Class' is the target variable
data = df.drop('target', axis=1)
prediction = loaded_model.predict(data)
print(prediction)
