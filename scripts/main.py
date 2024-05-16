import mlflow
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set the MLflow tracking URI to the server
mlflow.set_tracking_uri("http://127.0.0.1:5002/")

# Set or create an experiment
mlflow.set_experiment("Hyperparameter_Tuning_With_MLOps_Data")

# Load the dataset
df = pd.read_csv('./data/mlops_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Define the objective function to minimize
def objective(params):
    model_name = params['model_name']
    model_params = params['params']
    model = params['model'](random_state=42, **model_params)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model_params)
        mlflow.set_tag("model", model_name)  # Tag each run with the model name for easier filtering
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Some models might not support predict_proba
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]  # For ROC-AUC score
            roc_auc = roc_auc_score(y_test, probabilities)
            mlflow.log_metric("roc_auc", roc_auc)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        return {'loss': -accuracy, 'status': STATUS_OK}

# Define the parameter space for each model
space = hp.choice('classifier_type', [
    {
        'model_name': 'Logistic_Regression',
        'model': LogisticRegression,
        'params': {
            'C': hp.uniform('lr_C', 0.01, 10),
            'max_iter': hp.choice('lr_max_iter', [100, 200])
        }
    },
    {
        'model_name': 'Random_Forest',
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': hp.choice('rf_n_estimators', [100]),
            'max_depth': hp.choice('rf_max_depth', [None, 10, 20]),
            'min_samples_split': hp.choice('rf_min_samples_split', [2, 4, 6])
        }
    },
    {
        'model_name': 'Gradient_Boosting',
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': hp.choice('gb_n_estimators', [50, 100]),
            'learning_rate': hp.uniform('gb_learning_rate', 0.01, 0.2),
            'max_depth': hp.choice('gb_max_depth', [3, 5])
        }
    },
    {
        'model_name': 'SVM',
        'model': SVC,
        'params': {
            'C': hp.uniform('svm_C', 0.1, 10),
            'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
            'gamma': hp.uniform('svm_gamma', 0.01, 1)
        }
    },
    {
        'model_name': 'KNN',
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': hp.choice('knn_n_neighbors', [3, 5, 7]),
            'algorithm': hp.choice('knn_algorithm', ['auto', 'ball_tree'])
        }
    }
])

# Run the optimizer with fewer experiments
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=6, trials=trials)  # Reduced max_evals

print("Best parameters: ", best)
