# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_clean_data():
    """
    Paso 1: Load and clean the datasets
    """
    # Load train data
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    # Load test data
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    # Rename column
    train_data = train_data.rename(columns={"default payment next month": "default"})
    test_data = test_data.rename(columns={"default payment next month": "default"})

    # Remove ID column
    if "ID" in train_data.columns:
        train_data = train_data.drop(columns=["ID"])
    if "ID" in test_data.columns:
        test_data = test_data.drop(columns=["ID"])

    # Remove records with missing values
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Group EDUCATION values > 4 into "others" (4)
    train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return train_data, test_data


def split_data(train_data, test_data):
    """
    Paso 2: Split datasets into x and y
    """
    x_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]

    x_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """
    Paso 3: Create ML pipeline
    """
    # Identify categorical columns
    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]

    # Identify numerical columns
    numerical_columns = [col for col in x_train.columns if col not in categorical_columns]

    # Create preprocessor for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_columns),
            ("num", "passthrough", numerical_columns),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA()),  # Use all components by default
            ("scaler", StandardScaler()),  # Standardize features
            ("selector", SelectKBest(score_func=f_classif)),
            ("classifier", MLPClassifier(max_iter=1000, random_state=42)),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train, quick_mode=False):
    """
    Paso 4: Optimize hyperparameters using GridSearchCV
    """
    if quick_mode:
        # Optimized parameter grid for good performance with reasonable training time
        param_grid = {
            "pca__n_components": [20, 25],
            "selector__k": [20, 25],
            "classifier__hidden_layer_sizes": [(100,), (100, 50), (100, 100)],
            "classifier__activation": ["relu"],
            "classifier__alpha": [0.0001, 0.001],
        }
        cv_splits = 10  # Use 10-fold cross-validation as required
    else:
        # Full parameter grid for comprehensive search
        param_grid = {
            "pca__n_components": [15, 20, 25, 30],
            "selector__k": [15, 20, 25, 30],
            "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (150,)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01],
        }
        cv_splits = 10

    # Create GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    return grid_search


def save_model(model, filename="files/models/model.pkl.gz"):
    """
    Paso 5: Save model as compressed pickle
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6: Calculate metrics for train and test sets
    """
    metrics = []

    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Train metrics
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    }
    metrics.append(train_metrics)

    # Test metrics
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    }
    metrics.append(test_metrics)

    return metrics, y_train_pred, y_test_pred


def calculate_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred):
    """
    Paso 7: Calculate confusion matrices
    """
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Format confusion matrix for train
    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    # Format confusion matrix for test
    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    return [cm_train_dict, cm_test_dict]


def save_metrics(metrics, cm_matrices, filename="files/output/metrics.json"):
    """
    Save metrics to JSON file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
        for cm in cm_matrices:
            f.write(json.dumps(cm) + "\n")


def main(run_training=True, quick_mode=False):
    """
    Main execution function
    
    Args:
        run_training: If True, trains the model. If False, skips training.
        quick_mode: If True, uses minimal hyperparameters for faster training.
    """
    # Paso 1: Load and clean data
    print("Step 1: Loading and cleaning data...")
    train_data, test_data = load_and_clean_data()

    # Paso 2: Split data
    print("Step 2: Splitting data...")
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)

    # Paso 3: Create pipeline
    print("Step 3: Creating pipeline...")
    pipeline = create_pipeline(x_train)

    # Paso 4: Optimize hyperparameters
    print("Step 4: Optimizing hyperparameters...")
    grid_search = optimize_hyperparameters(pipeline, x_train, y_train, quick_mode=quick_mode)

    if run_training:
        # Train the model (COMPUTATIONALLY INTENSIVE)
        print("Step 4: Training model (this may take a while)...")
        grid_search.fit(x_train, y_train)

        # Paso 5: Save model
        print("Step 5: Saving model...")
        save_model(grid_search)

        # Paso 6 & 7: Calculate and save metrics
        print("Step 6 & 7: Calculating and saving metrics...")
        metrics, y_train_pred, y_test_pred = calculate_metrics(grid_search, x_train, y_train, x_test, y_test)
        cm_matrices = calculate_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred)
        save_metrics(metrics, cm_matrices)

        print("Done!")
    else:
        print("Training skipped. Set run_training=True to train the model.")


if __name__ == "__main__":
    # For full training with extensive hyperparameter search, use:
    # main(run_training=True, quick_mode=False)
    
    # For quick training with minimal hyperparameters, use:
    main(run_training=True, quick_mode=True)
    
    # To skip training entirely, use:
    # main(run_training=False)
