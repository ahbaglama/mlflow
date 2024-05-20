import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5002/")

mlflow.set_experiment("MLops_New_Dataset_Experiment")

df = pd.read_csv('../data/mlops_data.csv')

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# List of models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=100),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42), 
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        predictions = model.predict(X_test)
        # Handle models that do not support probability estimates
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1] 
            roc_auc = roc_auc_score(y_test, probabilities)
            mlflow.log_metric("roc_auc", roc_auc)
        
        # Calculate and log metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, f"model_{name.replace(' ', '_').lower()}")

print("Model training and logging completed.")
