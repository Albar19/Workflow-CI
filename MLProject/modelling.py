import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

def train_and_tune():
    # -------------------------------------------------------------------------
    # BYPASS OAUTH BROWSER: Mencegah error timeout/hanging saat jabat tangan SSL
    # -------------------------------------------------------------------------
    os.environ['DAGSHUB_USER_TOKEN'] = '57699fa27b07e650ee0590c85c42464b43fa2ab0'
    
    # Inisialisasi Koneksi MLflow ke DagsHub Online (Syarat Mutlak Advance)
    dagshub.init(repo_owner='Albar19', repo_name='Eksperimen_SML_Muhamad_Hafizh_Albar', mlflow=True)
    
    # 2. Sinkronisasi Jalur Dataset Preprocessing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'namadataset_preprocessing', 'clean_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}. Pastikan data hasil Kriteria 1 sudah disalin.")
        
    df = pd.read_csv(data_path)
    
    # Pisahkan fitur dan target (Ganti 'target' sesuai nama kolom target riil Anda jika berbeda)
    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # FIX BUG: Mengubah 'test_test_split=0.2' menjadi 'test_size=0.2'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter untuk kebutuhan manual logging tuning
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    mlflow.set_experiment("Tuning_Model_Churn")
    
    # 3. Eksekusi Training dan Manual Logging (Dilarang menggunakan autolog)
    with mlflow.start_run(run_name="RandomForest_Manual_Tuning"):
        # Manual Logging Parameter
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Training Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Inferensi Evaluasi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        
        # Manual Logging Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log Model Sklearn ke Tracking Server
        mlflow.sklearn.log_model(model, "model")
        
        # -------------------------------------------------------------------------
        # ARTEFAK TAMBAHAN 1: Plot Confusion Matrix
        # -------------------------------------------------------------------------
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        cm_plot_path = os.path.join(base_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path)
        plt.close()
        
        mlflow.log_artifact(cm_plot_path)
        if os.path.exists(cm_plot_path):
            os.remove(cm_plot_path)
            
        # -------------------------------------------------------------------------
        # ARTEFAK TAMBAHAN 2: Plot ROC Curve
        # -------------------------------------------------------------------------
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            
            roc_plot_path = os.path.join(base_dir, "roc_curve.png")
            plt.savefig(roc_plot_path)
            plt.close()
            
            mlflow.log_artifact(roc_plot_path)
            if os.path.exists(roc_plot_path):
                os.remove(roc_plot_path)
                
        print("Model berhasil dilatih. Parameter, metrik, dan 2 artefak sukses diunggah ke DagsHub!")

if __name__ == "__main__":
    train_and_tune()