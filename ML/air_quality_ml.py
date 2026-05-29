import os
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(CURRENT_DIR, "artifacts")
REPORT_DIR = os.path.join(CURRENT_DIR, "reports")
MONITOR_DIR = os.path.join(CURRENT_DIR, "monitoring")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "season", "weekday", "temp", "hum", "mq7", "mq135",
    "ky028_analog", "ky028_digital", "bmp_temp", "pressure",
    "altitude", "aht21_temp", "aht21_hum", "ens_iaq", "ens_tvoc", "ens_co2"
]

TARGET_COLUMN = "iaq_class"

DEFAULT_MEANS = {
    "season": 2,
    "weekday": 3,
    "temp": 22.0,
    "hum": 45.0,
    "mq7": 120.0,
    "mq135": 110.0,
    "ky028_analog": 35.0,
    "ky028_digital": 0.0,
    "bmp_temp": 15.0,
    "pressure": 1013.0,
    "altitude": 82.0,
    "aht21_temp": 22.0,
    "aht21_hum": 45.0,
    "ens_iaq": 75.0,
    "ens_tvoc": 250.0,
    "ens_co2": 800.0,
}

FAST_MODE = True


@dataclass
class ModelMetadata:
    version: str
    trained_at: str
    features: list
    best_params: dict
    train_rows: int
    test_rows: int
    train_accuracy: float
    test_accuracy: float
    train_f1_macro: float
    test_f1_macro: float
    data_hash: str


class AirQualityPredictor:
    def __init__(
        self,
        model_path="artifacts/air_quality_model.joblib",
        metadata_path="artifacts/metadata.json",
    ):
        self.model_path = os.path.join(CURRENT_DIR, model_path)
        self.metadata_path = os.path.join(CURRENT_DIR, metadata_path)
        self.model = None
        self.metadata = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.load_model()

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]
        for col, default in DEFAULT_MEANS.items():
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(-1, np.nan).fillna(default)
        return X

    def _build_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    self.feature_columns,
                )
            ]
        )
        clf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("clf", clf),
        ])

    def _param_space(self):
        if FAST_MODE:
            return {
                "clf__n_estimators": [80, 120],
                "clf__max_depth": [10, 15],
                "clf__min_samples_split": [2, 4],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt"],
                "clf__bootstrap": [True],
            }
        return {
            "clf__n_estimators": [100, 150, 200],
            "clf__max_depth": [10, 15, 20],
            "clf__min_samples_split": [2, 4],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
            "clf__bootstrap": [True, False],
        }

    def _hash_df(self, df: pd.DataFrame) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_metadata(self):
        if self.metadata is None:
            return
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, ensure_ascii=False, indent=2)

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.metadata = self._load_metadata()
            return True
        except Exception:
            self.model = None
            self.metadata = None
            return False

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self._save_metadata()

    def get_recommendation(self, iaq_class):
        recs = {
            0: "✅ Отлично",
            1: "🟢 Хорошо",
            2: "🟡 Удовл.",
            3: "🟠 Плохо",
            4: "🔴 Очень плохо",
            5: "🚨 АВАРИЯ!",
        }
        return recs.get(int(iaq_class), "Неизвестно")

    def fit(self, df: pd.DataFrame, test_size=0.2, random_state=42, automl_iter=3):
        df = df.copy()

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Нет целевого столбца {TARGET_COLUMN}")

        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)

        X = self.preprocess_features(df[self.feature_columns])
        y = df[TARGET_COLUMN]

        stratify_arg = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        pipe = self._build_pipeline()

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self._param_space(),
            n_iter=automl_iter if FAST_MODE else 5,
            scoring="f1_macro",
            cv=2 if FAST_MODE else 3,
            verbose=1,
            n_jobs=1,
            random_state=random_state,
            refit=True,
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = self._hash_df(df[self.feature_columns + [TARGET_COLUMN]])

        self.metadata = ModelMetadata(
            version=version,
            trained_at=datetime.now().isoformat(),
            features=self.feature_columns,
            best_params=search.best_params_,
            train_rows=int(len(X_train)),
            test_rows=int(len(X_test)),
            train_accuracy=float(train_acc),
            test_accuracy=float(test_acc),
            train_f1_macro=float(train_f1),
            test_f1_macro=float(test_f1),
            data_hash=data_hash,
        )

        self.save_model()
        self._write_training_report(search.best_params_, y_test, y_test_pred, search.best_score_)
        self._log_experiment(search.best_params_, train_acc, test_acc, train_f1, test_f1, len(df))

        return {
            "status": "trained",
            "version": version,
            "best_params": search.best_params_,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_f1_macro": float(train_f1),
            "test_f1_macro": float(test_f1),
            "best_cv_score": float(search.best_score_),
            "mode": "FAST" if FAST_MODE else "FULL",
        }

    def _write_training_report(self, best_params, y_test, y_test_pred, best_cv_score):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(REPORT_DIR, f"train_report_{ts}.txt")
        json_path = os.path.join(REPORT_DIR, f"train_report_{ts}.json")

        report_text = classification_report(y_test, y_test_pred, zero_division=0)
        report_json = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("BEST_PARAMS\n")
            f.write(json.dumps(best_params, ensure_ascii=False, indent=2))
            f.write("\n\nBEST_CV_SCORE\n")
            f.write(str(best_cv_score))
            f.write("\n\nCLASSIFICATION_REPORT\n")
            f.write(report_text)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "best_cv_score": best_cv_score,
                    "classification_report": report_json,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _log_experiment(self, best_params, train_acc, test_acc, train_f1, test_f1, rows):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "rows": int(rows),
            "best_params": best_params,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_f1_macro": float(train_f1),
            "test_f1_macro": float(test_f1),
        }
        with open(os.path.join(MONITOR_DIR, "experiments.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"air_quality_{self.metadata.version}"):
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(
                        {
                            "train_accuracy": train_acc,
                            "test_accuracy": test_acc,
                            "train_f1_macro": train_f1,
                            "test_f1_macro": test_f1,
                        }
                    )
            except Exception:
                pass

    def _log_prediction(self, sensor_data, result):
        record = dict(sensor_data)
        record.update(
            {
                "iaq_class_pred": int(result["iaq_class"]),
                "max_proba": float(max(result["probabilities"].values())) if result["probabilities"] else 0.0,
                "timestamp": datetime.now().isoformat(),
            }
        )
        with open(os.path.join(MONITOR_DIR, "predictions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def predict(self, sensor_data):
        if self.model is None:
            return {"error": "Модель не обучена!"}

        payload = dict(sensor_data)
        for col in self.feature_columns:
            payload.setdefault(col, -1)

        df = pd.DataFrame([payload])
        X = self.preprocess_features(df)

        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        classes = self.model.classes_

        result = {
            "iaq_class": int(pred),
            "probabilities": {int(k): round(float(v), 3) for k, v in zip(classes, proba)},
            "recommendation": self.get_recommendation(pred),
        }

        self._log_prediction(payload, result)
        return result

    def retrain_from_df(self, df: pd.DataFrame):
        return self.fit(df)

    def retrain_from_db(self, db_config, limit=10000):
        import psycopg2

        conn = psycopg2.connect(**db_config)
        try:
            query = f"""
            SELECT season, weekday, temp, hum, mq7, mq135, ky028_analog, ky028_digital,
                   bmp_temp, pressure, altitude, aht21_temp, aht21_hum, ens_iaq, ens_tvoc, ens_co2,
                   iaq_class
            FROM sensor.data
            WHERE iaq_class IS NOT NULL
            ORDER BY created_at DESC
            LIMIT {int(limit)}
            """
            df = pd.read_sql_query(query, conn)
            if len(df) < 100:
                return {"error": "Недостаточно данных в БД"}
            return self.fit(df)
        finally:
            conn.close()

    def drift_report(self, current_df: pd.DataFrame, baseline_df: pd.DataFrame):
        rows = []
        for col in self.feature_columns:
            c = pd.to_numeric(current_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            b = pd.to_numeric(baseline_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

            if len(c) < 10 or len(b) < 10:
                continue

            bins = 10
            b_hist, bin_edges = np.histogram(b, bins=bins)
            c_hist, _ = np.histogram(c, bins=bin_edges)

            b_pct = b_hist / max(b_hist.sum(), 1)
            c_pct = c_hist / max(c_hist.sum(), 1)

            psi = 0.0
            for bp, cp in zip(b_pct, c_pct):
                bp = max(bp, 1e-6)
                cp = max(cp, 1e-6)
                psi += (cp - bp) * np.log(cp / bp)

            rows.append(
                {
                    "feature": col,
                    "baseline_mean": float(b.mean()),
                    "current_mean": float(c.mean()),
                    "mean_shift": float(c.mean() - b.mean()),
                    "psi": float(psi),
                }
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(rows).to_csv(os.path.join(MONITOR_DIR, f"drift_{ts}.csv"), index=False)
        return rows

    def health(self):
        meta = self.metadata if isinstance(self.metadata, dict) else (asdict(self.metadata) if self.metadata else None)
        return {
            "model_loaded": self.model is not None,
            "metadata": meta,
            "features": self.feature_columns,
            "fast_mode": FAST_MODE,
        }

    def generate_test_data(self, n=20, with_target=True):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "season": rng.integers(1, 5, size=n),
                "weekday": rng.integers(1, 8, size=n),
                "temp": rng.normal(22, 3, size=n),
                "hum": rng.normal(45, 10, size=n),
                "mq7": rng.normal(120, 30, size=n),
                "mq135": rng.normal(110, 25, size=n),
                "ky028_analog": rng.integers(0, 1024, size=n),
                "ky028_digital": rng.integers(0, 2, size=n),
                "bmp_temp": rng.normal(15, 4, size=n),
                "pressure": rng.normal(1013, 8, size=n),
                "altitude": rng.normal(82, 5, size=n),
                "aht21_temp": rng.normal(22, 3, size=n),
                "aht21_hum": rng.normal(45, 10, size=n),
                "ens_iaq": rng.integers(0, 500, size=n),
                "ens_tvoc": rng.integers(0, 1000, size=n),
                "ens_co2": rng.integers(300, 2000, size=n),
            }
        )
        if with_target:
            df["iaq_class"] = rng.integers(0, 6, size=n)
        return df


predictor = AirQualityPredictor()

def create_app():
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def api_predict():
        data = request.json or {}
        result = predictor.predict(data)
        return jsonify(result)

    @app.route('/retrain', methods=['POST'])
    def api_retrain():
        payload = request.json or {}
        if "df_path" in payload:
            df = pd.read_csv(payload["df_path"])
            result = predictor.retrain_from_df(df)
        else:
            result = predictor.retrain_from_db(
                payload.get("db_config", {}),
                payload.get("limit", 10000)
            )
        return jsonify(result)

    @app.route('/status', methods=['GET'])
    def status():
        return jsonify(predictor.health())

    @app.route('/monitoring/drift', methods=['POST'])
    def monitoring_drift():
        payload = request.json or {}
        current_df = pd.read_csv(payload["current_df_path"])
        baseline_df = pd.read_csv(payload["baseline_df_path"])
        return jsonify({"features": predictor.drift_report(current_df, baseline_df)})

    @app.route('/monitoring/health', methods=['GET'])
    def monitoring_health():
        return jsonify({
            "status": "ok",
            "model_loaded": predictor.model is not None,
            "timestamp": datetime.now().isoformat()
        })

    @app.route('/')
    def index():
        return """
        <!DOCTYPE html>
        <html><head><title>ML Air Quality</title>
        <style>body{font-family:Arial;padding:20px;} .endpoint{background:#f0f8ff;padding:10px;margin:10px 0;border-left:4px solid #007bff;}</style>
        </head><body>
        <h1>ML Air Quality Predictor</h1>
        <h2>Эндпоинты:</h2>
        <div class="endpoint">POST /predict — прогноз IAQ</div>
        <div class="endpoint">GET /status — статус</div>
        <div class="endpoint">POST /retrain — переобучение из БД или CSV</div>
        <div class="endpoint">POST /monitoring/drift — отчёт по дрейфу</div>
        <div class="endpoint">GET /monitoring/health — health-check</div>
        </body></html>
        """

    return app


if __name__ == "__main__":
    csv_filename = "air_quality_training_dataset_2.csv"
    csv_path = os.path.join(CURRENT_DIR, csv_filename)

    if os.path.exists(csv_path):
        print(f"Training from: {csv_path}")
        df = pd.read_csv(csv_path)
        result = predictor.fit(df)
        print(result)
    else:
        print(f"CSV not found: {csv_path}")

    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=False)