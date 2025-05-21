import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from src.features.mlflowsetup import MLflowConfig
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

class ModelEvaluator:
    def __init__(self, output_dir="evaluation_results", mlflow_experiment=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Centralized MLflow setup
        self.mlflow_config = MLflowConfig()
        self.mlflow_config.setup()

        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def evaluate_model(self, model, X_test, y_test, model_name="model"):
        with mlflow.start_run(run_name=model_name):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Basic metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            if y_proba is not None:
                auc = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("roc_auc", auc)
            else:
                auc = None

            # Save classification report
            report_path = f"{self.output_dir}/{model_name}_report.csv"
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            pd.DataFrame(report).transpose().to_csv(report_path)
            mlflow.log_artifact(report_path)

            # Save confusion matrix
            self._save_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(f"{self.output_dir}/{model_name}_conf_matrix.png")

            # Save ROC curve
            if y_proba is not None:
                self._save_roc_curve(y_test, y_proba, model_name)
                mlflow.log_artifact(f"{self.output_dir}/{model_name}_roc_curve.png")

            # Save feature importance if applicable
            if hasattr(model, 'feature_importances_') and isinstance(X_test, pd.DataFrame):
                self._save_feature_importance(model, X_test, model_name)
                mlflow.log_artifact(f"{self.output_dir}/{model_name}_feature_importance.csv")
                mlflow.log_artifact(f"{self.output_dir}/{model_name}_feature_importance.png")

            # Save summary CSV
            summary_path = f"{self.output_dir}/{model_name}_metrics_summary.csv"
            summary_df = pd.DataFrame([{
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc
            }])
            summary_df.to_csv(summary_path, index=False)
            mlflow.log_artifact(summary_path)

            # Set tags for better traceability
            mlflow.set_tags({
                "model_name": model_name,
                "run_type": "evaluation",
                "evaluator": "ModelEvaluator"
            })

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\nModel: {model_name}")
            print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
            if auc is not None:
                print(f"ROC AUC: {auc:.4f}")

            return y_pred, y_proba

    def evaluate_from_file(self, model_path, X_test, y_test):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model = joblib.load(model_path)
        print(f"Evaluating model from {model_path}")
        return self.evaluate_model(model, X_test, y_test, model_name)

    def _save_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_conf_matrix.png")
        plt.close()

    def _save_roc_curve(self, y_true, y_proba, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_roc_curve.png")
        plt.close()

    def _save_feature_importance(self, model, X, model_name):
        importance = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by="Importance", ascending=False)

        fi_df.to_csv(f"{self.output_dir}/{model_name}_feature_importance.csv", index=False)
        fi_df.head(20).plot(kind='bar', x='Feature', y='Importance', title='Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_feature_importance.png")
        plt.close()