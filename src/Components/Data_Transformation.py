import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

from src.Exception import CustomException
from src.Logger import logging
from src.Utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X: pd.DataFrame):
        """
        Create preprocessing pipelines for numerical and categorical features.
        """
        try:
            # Separate columns by datatype
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

            # Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Drop non-predictive/created columns
            drop_cols = ['customerID', 'tenure_bin', 'monthly_charges_bin']
            train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True, errors="ignore")
            test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True, errors="ignore")

            # Encode target
            le = LabelEncoder()
            train_df['Churn'] = le.fit_transform(train_df['Churn'])
            test_df['Churn'] = le.transform(test_df['Churn'])

            target_column_name = "Churn"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Add clustering as extra feature
            kmeans = KMeans(n_clusters=3, random_state=42)
            train_clusters = kmeans.fit_predict(input_feature_train_arr)
            test_clusters = kmeans.predict(input_feature_test_arr)

            input_feature_train_arr = np.hstack([input_feature_train_arr, train_clusters.reshape(-1, 1)])
            input_feature_test_arr = np.hstack([input_feature_test_arr, test_clusters.reshape(-1, 1)])

            # Combine with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
