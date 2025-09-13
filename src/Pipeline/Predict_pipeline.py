import sys
import os
import numpy as np
import pandas as pd
from src.Exception import CustomException
from src.Utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            kmeans_path = os.path.join("artifacts", "kmeans.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            kmeans = load_object(file_path=kmeans_path)

            # Step 1: Preprocess features (scaling + encoding)
            data_scaled = preprocessor.transform(features)

            # Step 2: Predict cluster
            cluster_label = kmeans.predict(data_scaled)
            data_with_cluster = np.hstack([data_scaled, cluster_label.reshape(-1, 1)])

            # Step 3: Predict final churn
            preds = model.predict(data_with_cluster)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 tenure: int,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float):

        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
