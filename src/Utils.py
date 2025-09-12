import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.Exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, scoring="f1", n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # set best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # properly train

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # collect metrics
            report[model_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_prob),
            }

            trained_models[model_name] = model  # store fitted model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)