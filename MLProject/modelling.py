"""
Model Training Script for MLflow Project (Workflow-CI).

Script ini dijalankan via `mlflow run .` di dalam folder MLProject.
Mendukung:
- Local MLflow tracking (default)
- DagsHub remote tracking (via env vars)
- Menyimpan artefak lokal di artifacts/

Dataset: Telco Customer Churn (preprocessed)
Author: Muhamad Hafizh Albar
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    import dagshub
except ImportError:
    dagshub = None


def _configure_mlflow() -> None:
    """Konfigurasi MLflow tracking URI dan DagsHub (jika tersedia)."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    dagshub_owner = os.environ.get("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.environ.get("DAGSHUB_REPO_NAME")
    dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")

    if dagshub is not None and dagshub_owner and dagshub_repo and dagshub_token:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)


def train_and_tune() -> None:
    """Training model dengan manual logging dan artefak."""
    _configure_mlflow()

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "namadataset_preprocessing" / "clean_data.csv"
    artifact_dir = base_dir / "artifacts"
    plots_dir = artifact_dir / "plots"
    model_dir = artifact_dir / "model"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {data_path}. "
            "Pastikan data hasil Kriteria 1 sudah disalin."
        )

    df = pd.read_csv(data_path)
    print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Pisahkan fitur dan target
    target_column = "Churn"
    X = df.drop(columns=[target_column])
    y = df[target_column]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameters
    n_estimators = 150
    max_depth = 12
    min_samples_split = 3
    random_state = 42

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Tuning_Model_Churn")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_CI_Retrain") as run:
        # Manual Logging Parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", random_state)

        # Training
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # Inferensi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")

        # Manual Logging Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log model to MLflow & save locally
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, str(model_dir))

        # Save metrics JSON
        metrics_path = artifact_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(metrics_path))

        # Artefak: Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_plot_path = plots_dir / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(cm_plot_path))

        # Artefak: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_plot_path = plots_dir / "roc_curve.png"
        plt.savefig(roc_plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(roc_plot_path))

        # Artefak: Feature Importance
        importances = model.feature_importances_
        top_n = min(15, len(feature_names))
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices][::-1], align="center", color="steelblue")
        plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        fi_plot_path = plots_dir / "feature_importance.png"
        plt.savefig(fi_plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(fi_plot_path))

        # Artefak: Classification Report JSON
        report = classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        )
        cr_path = artifact_dir / "classification_report.json"
        cr_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cr_path))

        print(f"Model berhasil dilatih. Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Model lokal tersimpan di: {model_dir}")


if __name__ == "__main__":
    train_and_tune()