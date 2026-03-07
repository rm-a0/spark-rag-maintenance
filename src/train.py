import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.config import settings
from src.pipeline import get_spark

_EXCLUDE = {"engine_id", "cycle", "max_cycle", "RUL", "failure_soon"}


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read the Parquet produced by pipeline.py and return numpy arrays."""
    with get_spark() as spark:
        pdf = spark.read.parquet(str(settings.parquet_path)).toPandas()

    feature_cols = [c for c in pdf.columns if c not in _EXCLUDE]
    X     = pdf[feature_cols].values
    y_rul = pdf["RUL"].values
    y_cls = pdf["failure_soon"].values

    print(f"Loaded {len(X):,} rows, {len(feature_cols)} features")
    return X, y_rul, y_cls, feature_cols


def train_regressor(X_tr, y_tr, X_val, y_val) -> xgb.XGBRegressor:
    """Predict exact RUL (continuous output)."""
    cfg = settings.xgb_regressor
    model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"Regressor  RMSE : {rmse:.2f} cycles")
    return model


def train_classifier(X_tr, y_tr, X_val, y_val) -> xgb.XGBClassifier:
    """
    Predict failure probability within the next failure_cycles cycles (binary).
    Evaluated with ROC-AUC — insensitive to class imbalance, unlike accuracy.
    """
    cfg              = settings.xgb_classifier
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Classifier ROC-AUC : {auc:.4f}")
    return model


def run_training() -> None:
    """Full training run."""
    X, y_rul, y_cls, feature_cols = load_features()

    X_tr, X_val, y_rul_tr, y_rul_val, y_cls_tr, y_cls_val = train_test_split(
        X, y_rul, y_cls,
        test_size=0.2,
        random_state=settings.xgb_regressor.random_state,
    )

    print("\n--- Regressor ---")
    reg = train_regressor(X_tr, y_rul_tr, X_val, y_rul_val)

    print("\n--- Classifier ---")
    cls = train_classifier(X_tr, y_cls_tr, X_val, y_cls_val)

    settings.artefacts_dir.mkdir(parents=True, exist_ok=True)

    reg.save_model(str(settings.artefacts_dir / "xgb_rul.json"))
    cls.save_model(str(settings.artefacts_dir / "xgb_failure_cls.json"))

    settings.feature_cols_path.write_text(json.dumps(feature_cols, indent=2))

    print(f"\nArtefacts saved → {settings.artefacts_dir}/")