import os
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
import joblib

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def fetch_datasets():
    print("Fetching datasets...")
    # 1. Breast Cancer (from sklearn)
    bc = load_breast_cancer()
    bc_df = pd.DataFrame(bc.data, columns=bc.feature_names)
    bc_df['target'] = bc.target
    bc_df.to_csv('data/breast_cancer.csv', index=False)
    
    # 2. Heart Disease (UCI)
    heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        urllib.request.urlretrieve(heart_url, "data/heart_disease.csv")
        heart_df = pd.read_csv("data/heart_disease.csv", names=heart_cols, na_values='?')
        # Convert target > 0 to 1 (binary classification: 0 = no disease, 1 = disease)
        heart_df['target'] = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)
        heart_df.to_csv('data/heart_disease.csv', index=False)
    except Exception as e:
        print(f"Error fetching Heart Disease data: {e}")

    # 3. Diabetes (Pima Indians)
    diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'target']
    try:
        urllib.request.urlretrieve(diabetes_url, "data/diabetes.csv")
        diabetes_df = pd.read_csv("data/diabetes.csv", names=diabetes_cols)
        diabetes_df.to_csv('data/diabetes.csv', index=False)
    except Exception as e:
        print(f"Error fetching Diabetes data: {e}")
        
    print("Datasets fetched and saved.")

def train_and_evaluate(dataset_name, df, target_col='target'):
    print(f"\n--- Training Models for {dataset_name} ---")
    
    # Drop rows where target is missing just in case
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = -1
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc,
            'ConfusionMatrix': cm.tolist()
        }
        
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        # We will save all models, but keep track of best f1 to select default
        # Save model
        joblib.dump(model, f'models/{dataset_name}_{name.replace(" ", "")}.pkl')
        
    # Save artifacts
    joblib.dump({
        'imputer': imputer,
        'scaler': scaler,
        'features': list(X.columns),
        'results': results
    }, f'models/{dataset_name}_metadata.pkl')

    print(f"Finished {dataset_name}. Metadata saved.")

def main():
    fetch_datasets()
    
    if os.path.exists('data/breast_cancer.csv'):
        df_bc = pd.read_csv('data/breast_cancer.csv')
        train_and_evaluate('BreastCancer', df_bc)
        
    if os.path.exists('data/heart_disease.csv'):
        df_hd = pd.read_csv('data/heart_disease.csv')
        train_and_evaluate('HeartDisease', df_hd)
        
    if os.path.exists('data/diabetes.csv'):
        df_diab = pd.read_csv('data/diabetes.csv')
        train_and_evaluate('Diabetes', df_diab)

if __name__ == "__main__":
    main()
