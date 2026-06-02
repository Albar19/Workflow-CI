import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split

try:
    import dagshub
except ImportError:
    dagshub = None


def _configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    dagshub_owner = os.environ.get("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.environ.get("DAGSHUB_REPO_NAME")
    dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")

    if dagshub is not None and dagshub_owner and dagshub_repo and dagshub_token:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)

def train_and_tune():
    _configure_mlflow()

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'namadataset_preprocessing' / 'clean_data.csv'
    artifact_dir = base_dir / 'artifacts'
    plots_dir = artifact_dir / 'plots'
    model_dir = artifact_dir / 'model'
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}. Pastikan data hasil Kriteria 1 sudah disalin.")

    df = pd.read_csv(data_path)

    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = 100
    max_depth = 10
    random_state = 42

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Tuning_Model_Churn")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Manual_Tuning") as run:
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, str(model_dir))

        metrics_path = artifact_dir / 'metrics.json'
        metrics_path.write_text(
            json.dumps(
                {
                    'run_id': run.info.run_id,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                },
                indent=2,
            ),
            encoding='utf-8',
        )
        mlflow.log_artifact(str(metrics_path))

        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        cm_plot_path = plots_dir / 'confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()

        mlflow.log_artifact(str(cm_plot_path))

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

            roc_plot_path = plots_dir / 'roc_curve.png'
            plt.savefig(roc_plot_path)
            plt.close()

            mlflow.log_artifact(str(roc_plot_path))

        print(f"Model berhasil dilatih. Run ID: {run.info.run_id}")
        print(f"Model lokal tersimpan di: {model_dir}")


if __name__ == "__main__":
    train_and_tune()