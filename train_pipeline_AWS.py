import joblib
import os
import re
import warnings
import argparse
import json
import tempfile
from datetime import datetime
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import boto3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings('ignore')

class DataLoader:
    DROPPED_COLS = ['Number', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Type_of_Loan']
    TARGET_COL   = 'Credit_Score'

    def __init__(self, filepath: str, aws_region: str = None, aws_profile: str = None):
        self.filepath    = filepath
        self.aws_region  = aws_region
        self.aws_profile = aws_profile
        self.data: pd.DataFrame = None

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        return str(path).startswith('s3://')

    def _download_from_s3(self, s3_uri: str) -> str:
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key    = parsed.path.lstrip('/')

        session_kwargs = {}
        if self.aws_profile:
            session_kwargs['profile_name'] = self.aws_profile
        session = boto3.session.Session(**session_kwargs)
        s3 = session.client('s3', region_name=self.aws_region)

        suffix = os.path.splitext(key)[1] or '.csv'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()

        print(f"[DataLoader] Mengunduh s3://{bucket}/{key} -> {tmp.name}")
        s3.download_file(bucket, key, tmp.name)
        return tmp.name

    def load(self) -> pd.DataFrame:
        print(f"[DataLoader] Memuat data dari: {self.filepath}")

        if self._is_s3_path(self.filepath):
            local_path = self._download_from_s3(self.filepath)
        else:
            local_path = self.filepath

        df = pd.read_csv(local_path)

        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Kolom target '{self.TARGET_COL}' tidak ditemukan. "
                             f"Kolom yang tersedia: {df.columns.tolist()}")

        print(f"[DataLoader] Shape awal: {df.shape}")
        df = self._drop_irrelevant(df)
        df = self._clean(df)
        self.data = df
        print(f"[DataLoader] Shape setelah cleaning: {df.shape}")
        print(f"[DataLoader] Missing values:\n{df.isnull().sum()}\n")
        return df

    def _drop_irrelevant(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_exist = [c for c in self.DROPPED_COLS if c in df.columns]
        return df.drop(columns=cols_exist)

    @staticmethod
    def _parse_credit_age(val):
        if pd.isna(val):
            return np.nan
        years  = re.search(r'(\d+)\s*Year',  str(val))
        months = re.search(r'(\d+)\s*Month', str(val))
        total  = 0
        if years:  total += int(years.group(1))  * 12
        if months: total += int(months.group(1))
        return total if total > 0 else np.nan

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Credit_History_Age' in df.columns:
            df['Credit_History_Months'] = df['Credit_History_Age'].apply(self._parse_credit_age)
            df = df.drop(columns=['Credit_History_Age'])

        if 'Num_of_Loan' in df.columns:
            df['Num_of_Loan'] = (
                df['Num_of_Loan'].astype(str)
                .str.replace('_', '', regex=False)
            )
            df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce')
            df['Num_of_Loan'] = df['Num_of_Loan'].clip(lower=0)

        if 'Occupation' in df.columns:
            df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
        if 'Payment_of_Min_Amount' in df.columns:
            df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'Unknown')
        if 'Changed_Credit_Limit' in df.columns:
            df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].replace('_', np.nan)
        if 'Payment_Behaviour' in df.columns:
            df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
            payment_mode = df['Payment_Behaviour'].mode()[0]
            df['Payment_Behaviour'] = df['Payment_Behaviour'].fillna(payment_mode)

        numeric_to_convert = [
            'Annual_Income', 'Age', 'Changed_Credit_Limit',
            'Outstanding_Debt', 'Amount_invested_monthly',
            'Num_of_Delayed_Payment', 'credit_mixes'
        ]
        for col in numeric_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


class Preprocessor:
    IQR_IMPUTE_COLS = [
        'Monthly_Inhand_Salary', 'Amount_invested_monthly', 'Monthly_Balance',
        'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_History_Months',
        'Outstanding_Debt', 'Age', 'Annual_Income', 'Changed_Credit_Limit', 'credit_mixes'
    ]

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state

        self.impute_values: dict        = {}
        self.encoders: dict             = {}
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.scaler: StandardScaler     = StandardScaler()
        self.num_cols: list             = []
        self.cat_cols: list             = []

    @staticmethod
    def _iqr_robust_median(series: pd.Series) -> float:
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR    = Q3 - Q1
        mask   = (series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)
        return series[mask].median()

    def _fit_imputer(self, X_train: pd.DataFrame):
        for col in self.IQR_IMPUTE_COLS:
            if col in X_train.columns:
                median = self._iqr_robust_median(X_train[col].dropna())
                self.impute_values[col] = median

    def _apply_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, val in self.impute_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X

    def _fit_encoders(self, X_train: pd.DataFrame):
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            self.encoders[col] = le

    def _apply_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = X[col].astype(str).apply(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            X[col] = le.transform(X[col])
        return X

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Credit_Score'):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        print(f"[Preprocessor] Numerical features ({len(self.num_cols)}): {self.num_cols}")
        print(f"[Preprocessor] Categorical features ({len(self.cat_cols)}): {self.cat_cols}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        self._fit_imputer(X_train)
        X_train = self._apply_imputer(X_train)
        X_test  = self._apply_imputer(X_test)

        self._fit_encoders(X_train)
        X_train = self._apply_encoders(X_train)
        X_test  = self._apply_encoders(X_test)

        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc  = self.label_encoder.transform(y_test)
        print(f"[Preprocessor] Class mapping: "
              f"{dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        X_train[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols])
        X_test[self.num_cols]  = self.scaler.transform(X_test[self.num_cols])

        print(f"[Preprocessor] X_train: {X_train.shape} | X_test: {X_test.shape}\n")
        return X_train, X_test, y_train_enc, y_test_enc

    def get_params(self) -> dict:
        """Return impute values untuk di-log ke MLflow."""
        return {f"impute_{k}": round(v, 4) for k, v in self.impute_values.items()}


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models: dict = {}
        self._model_registry = self._build_registry()

    def _build_registry(self) -> dict:
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, use_label_encoder=False,
                eval_metric='mlogloss', random_state=self.random_state
            ),
        }

    def get_model_names(self) -> list:
        return list(self._model_registry.keys())

    def train(self, name: str, X_train: pd.DataFrame, y_train: np.ndarray):
        """Train satu model berdasarkan nama."""
        if name not in self._model_registry:
            raise ValueError(f"Model '{name}' tidak ditemukan dalam registry.")
        print(f"[ModelTrainer] Training: {name}...")
        model = self._model_registry[name]
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        print(f"[ModelTrainer] {name} done.\n")
        return model

    def train_all(self, X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
        """Train semua model dalam registry."""
        for name in self._model_registry:
            self.train(name, X_train, y_train)
        return self.trained_models

    def get_model_params(self, name: str) -> dict:
        """Return hyperparameter model untuk logging."""
        model = self._model_registry.get(name)
        if model is None:
            return {}
        return model.get_params()


class ModelEvaluator:
    def __init__(self, class_names: list, output_dir: str = "mlflow_artifacts"):
        self.class_names = class_names
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: dict = {}

    def evaluate(
        self,
        name: str,
        model,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feature_names: list = None
    ) -> dict:
        """Evaluasi satu model, return dict metrik."""
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, target_names=self.class_names)

        metrics = {
            'accuracy':        round(acc, 4),
            'f1_macro':     round(f1,  4),
        }
        self.results[name] = metrics

        print(f"[ModelEvaluator] {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 (wtd) : {f1:.4f}")
        print(f"  Classification Report:\n{report}")

        return metrics, y_pred, report

    def plot_confusion_matrix(self, name: str, y_test: np.ndarray, y_pred: np.ndarray) -> str:
        cm   = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()

        safe_name = name.replace(' ', '_')
        path = os.path.join(self.output_dir, f"cm_{safe_name}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def plot_feature_importance(self, name: str, model, feature_names: list) -> str | None:
        if not hasattr(model, 'feature_importances_'):
            return None
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind='bar', color='teal', ax=ax)
        ax.set_title(f'Top 15 Feature Importances — {name}')
        ax.set_ylabel('Importance')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        safe_name = name.replace(' ', '_')
        path = os.path.join(self.output_dir, f"fi_{safe_name}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def plot_comparison(self) -> str:
        results_df = pd.DataFrame(self.results).T

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        results_df['accuracy'].sort_values().plot(
            kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_xlabel('Accuracy')
        axes[0].set_xlim(0, 1)

        results_df['f1_macro'].sort_values().plot(
            kind='barh', ax=axes[1], color='coral')
        axes[1].set_title('Model F1-Score (Macro) Comparison')
        axes[1].set_xlabel('F1-Score')
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"[ModelEvaluator] Comparison chart disimpan: {path}")
        return path

    def get_best_model_name(self, metric: str = 'f1_macro') -> str:
        return max(self.results, key=lambda k: self.results[k][metric])

class Pipeline:
    EXPERIMENT_NAME = "Credit_Score_Classification"

    def __init__(
        self,
        data_path:    str   = "data_C.csv",
        test_size:    float = 0.2,
        random_state: int   = 42,
        mlflow_uri:   str   = "sqlite:///mlflow.db",
        aws_region:   str   = None,
        aws_profile:  str   = None,
    ):
        self.data_path    = data_path
        self.test_size    = test_size
        self.random_state = random_state
        self.aws_region   = aws_region
        self.aws_profile  = aws_profile

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.EXPERIMENT_NAME)
        print(f"[Pipeline] MLflow tracking URI : {mlflow_uri}")
        print(f"[Pipeline] Experiment           : {self.EXPERIMENT_NAME}\n")

    def run(self):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        loader = DataLoader(self.data_path, aws_region=self.aws_region, aws_profile=self.aws_profile)
        df     = loader.load()

        preprocessor = Preprocessor(self.test_size, self.random_state)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        class_names    = list(preprocessor.label_encoder.classes_)
        feature_names  = X_train.columns.tolist()
        os.makedirs("model_artifacts", exist_ok=True)
        joblib.dump(preprocessor.scaler,        "model_artifacts/scaler.pkl")
        joblib.dump(preprocessor.encoders,      "model_artifacts/encoders.pkl")
        joblib.dump(preprocessor.impute_values, "model_artifacts/impute_values.pkl")
        joblib.dump(preprocessor.num_cols,      "model_artifacts/num_cols.pkl")
        joblib.dump(preprocessor.cat_cols,      "model_artifacts/cat_cols.pkl")
        print("[Pipeline] Preprocessor artifacts saved to model_artifacts/")

        trainer   = ModelTrainer(self.random_state)
        trainer.train_all(X_train, y_train)

        evaluator = ModelEvaluator(class_names)

        for name, model in trainer.trained_models.items():
            safe_name = name.replace(' ', '_')
            run_name  = f"{safe_name}_{run_timestamp}"

            with mlflow.start_run(run_name=run_name):

                mlflow.log_param("model_name",    name)
                mlflow.log_param("test_size",     self.test_size)
                mlflow.log_param("random_state",  self.random_state)
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples",  X_test.shape[0])
                mlflow.log_param("n_features",    X_train.shape[1])

                for k, v in trainer.get_model_params(name).items():
                    mlflow.log_param(f"hp_{k}", v)

                for k, v in preprocessor.get_params().items():
                    mlflow.log_param(k, v)

                metrics, y_pred, report = evaluator.evaluate(
                    name, model, X_test, y_test, feature_names
                )

                mlflow.log_metrics(metrics)

                report_path = os.path.join(
                    evaluator.output_dir, f"report_{safe_name}.txt"
                )
                with open(report_path, 'w') as f:
                    f.write(f"Model: {name}\n\n{report}")
                mlflow.log_artifact(report_path)

                cm_path = evaluator.plot_confusion_matrix(name, y_test, y_pred)
                mlflow.log_artifact(cm_path)

                fi_path = evaluator.plot_feature_importance(name, model, feature_names)
                if fi_path:
                    mlflow.log_artifact(fi_path)

                if 'XGBoost' in name:
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"[MLflow] Run '{run_name}' done di-log.\n")

        comparison_path = evaluator.plot_comparison()

        best_name    = evaluator.get_best_model_name()
        best_model_obj = trainer.trained_models[best_name]
        best_metrics = evaluator.results[best_name]
        joblib.dump(best_model_obj, "model_artifacts/best_model.pkl")
        print(f"[Pipeline] Saved best model ({best_name}) to model_artifacts/best_model.pkl")
        with mlflow.start_run(run_name=f"experiment_summary_{run_timestamp}"):
            mlflow.log_artifact(comparison_path)

            all_results_path = os.path.join(evaluator.output_dir, "all_results.json")
            with open(all_results_path, 'w') as f:
                json.dump(evaluator.results, f, indent=2)
            mlflow.log_artifact(all_results_path)

            mlflow.log_param("best_model", best_name)
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

            if 'XGBoost' in best_name:
                mlflow.xgboost.log_model(
                    best_model_obj,
                    name="best_model",
                    registered_model_name="CreditScoreClassifier"
                )
            else:
                mlflow.sklearn.log_model(
                    best_model_obj,
                    name="best_model",
                    registered_model_name="CreditScoreClassifier"
                )
            print(f"[MLflow] Best model '{best_name}' registered as 'CreditScoreClassifier'")

        print("\n" + "="*55)
        print("  EXPERIMENT RESULTS SUMMARY")
        print("="*55)
        results_df = pd.DataFrame(evaluator.results).T.sort_values('f1_macro', ascending=False)
        print(results_df.to_string())
        print("="*55)
        print(f"\n Best Model  : {best_name}")
        print(f"   Accuracy    : {best_metrics['accuracy']:.4f}")
        print(f"   F1 (wtd)    : {best_metrics['f1_macro']:.4f}")
        print("\n[Pipeline] done.")
        print("  mlflow ui --backend-store-uri mlruns\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Score Classification Pipeline")
    parser.add_argument("--data",         type=str,   default="data_C.csv",
                        help="Path ke file CSV dataset. Bisa path lokal atau S3 URI "
                             "(contoh: s3://my-bucket/data/data_C.csv)")
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Proporsi test set (default: 0.2)")
    parser.add_argument("--random-state", type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--mlflow-uri",   type=str,   default="sqlite:///mlflow.db",
                        help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--aws-region",   type=str,   default=None,
                        help="AWS region tempat bucket S3 berada (opsional, contoh: ap-southeast-1)")
    parser.add_argument("--aws-profile",  type=str,   default=None,
                        help="Nama AWS CLI profile untuk dev lokal. Di EC2 biarkan kosong, "
                             "kredensial otomatis diambil dari IAM Role instance.")
    args = parser.parse_args()

    pipeline = Pipeline(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        mlflow_uri=args.mlflow_uri,
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
    )
    pipeline.run()