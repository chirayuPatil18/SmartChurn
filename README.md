📌 SmartChurn: Customer Churn Prediction with Clustering + Classification
📖 Project Overview

SmartChurn is an end-to-end Machine Learning pipeline designed to predict customer churn with enhanced interpretability.
Unlike standard churn models, SmartChurn combines:

Unsupervised Clustering (KMeans) → groups customers by behavioral patterns.

Supervised Classification (Random Forest) → predicts churn probability within each cluster.

This two-step hybrid approach allows the model to adapt churn prediction rules to different customer segments (e.g., long-term loyal customers vs. high-risk short-term users).

🏗️ Key Features

Modular Project Structure with separate modules for data ingestion, preprocessing, model training, logging, and exception handling.

Robust Data Preprocessing

Handles missing values, categorical encoding, and feature scaling.

Automatic pipelines via ColumnTransformer.

Clustering + Classification Hybrid

KMeans for unsupervised customer segmentation.

RandomForestClassifier for churn prediction.

Class Imbalance Handling

Tested both class_weight="balanced" and SMOTE oversampling.

Visualization & EDA

Customer behavior analysis (tenure, monthly charges, contract type).

Correlation heatmaps, churn rate per cluster, PCA cluster visualization.

Deployment Ready

Jupyter Notebook for EDA & model development.

Modular Python package for training & prediction.

Flask-based API (extendable to AWS, Azure, or Docker).

📂 Project Structure
SmartChurn/
|-- README.md
|-- requirements.txt
|-- setup.py
|-- .gitignore
|-- application.py
|-- Procfile               
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
|   └── data/
|        └── raw                      
|        └── processed
├── src/                          # package source (importable)
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── Utils.py                 
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│       ├── __init__.py
│       └── predict_pipeline.py
|-- temlates/
|   └── home.html
|   └── index.html
├── artifacts/                     # model artifacts (preprocessor.pkl, model.pkl, cluster.pkl)
├── logs/                          # runtime logs


⚙️ Tech Stack

Language: Python 3.9+

Libraries:

Data: pandas, numpy, seaborn, matplotlib

ML: scikit-learn, imblearn

Deployment: Flask (future-ready for AWS/Azure)

Environment: Virtual Environment (venv) / Jupyter Notebook

Version Control: Git & GitHub

🚀 How It Works

Data Ingestion → Reads and validates raw customer churn data.

Preprocessing Pipeline → Encodes categorical features, scales numerical features.

Clustering (KMeans) → Assigns customers to behavior-based groups.

Classification (Random Forest) → Predicts churn probability using cluster info.

Evaluation → Reports accuracy, precision, recall, F1-score, ROC-AUC.

Deployment → Exposes prediction pipeline via Flask API (future-ready for cloud).

📊 Results

Improved Recall for churn class after applying SMOTE.

Cluster-aware model outperformed baseline Random Forest.

Churn Profiles Identified:

Cluster 0 → Stable long-tenure customers (low churn).

Cluster 1 → High-risk short-term, high monthly charges (high churn).

Cluster 2 → Mid-risk customers with mixed profiles.

🔮 Future Improvements

Integrate SHAP/LIME for feature interpretability.

Automate hyperparameter tuning with GridSearchCV / Optuna.

Deploy fully on AWS Elastic Beanstalk / Azure Container Apps.

Extend to streaming predictions (Kafka + Spark) for real-time churn monitoring.

👨‍💻 Author

Developed by Chirayu Patil as part of an end-to-end ML engineering portfolio project.