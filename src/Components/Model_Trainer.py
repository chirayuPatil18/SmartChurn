import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.Exception import CustomException
from src.Logger import logging
from src.Utils import save_object, evaluate_models

import numpy as np


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Handle class imbalance
            num_pos = np.sum(y_train == 1)  # churners
            num_neg = np.sum(y_train == 0)  # non-churners
            scale_pos_weight = num_neg / num_pos
            logging.info(f"Calculated scale_pos_weight: {scale_pos_weight}")

            # Models dictionary
            models = {
                "XGBoost Classifier": XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42
                )
            }

            # Hyperparameter grid
            params = {
                "XGBoost Classifier": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'colsample_bytree': [0.7, 0.8, 1.0],
                    'subsample': [0.7, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                }
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Pick best model
            best_model_name = max(model_report, key=lambda k: model_report[k]["F1"])
            best_model = trained_models[best_model_name]   # ✅ now it's fitted
            best_score = model_report[best_model_name]["F1"]


            if best_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model found: {best_model_name} with F1 score: {best_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate on test
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
            }

            logging.info(f"Final Test Metrics: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)
