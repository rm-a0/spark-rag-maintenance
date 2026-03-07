import json
import numpy as np
import xgboost as xgb
from dataclasses import dataclass

from src.config import settings
from src.pipeline import get_spark


@dataclass
class RiskReport:
    engine_id:       int
    rul_pred:        float
    failure_prob:    float
    risk_level:      str          # HIGH / MEDIUM / LOW
    sensor_snapshot: dict[str, float]
    recommendation:  str | None = None

    def print(self):
        print("\n" + "=" * 50)
        print(f"  TURBINE SENTINEL — ENGINE #{self.engine_id}")
        print("=" * 50)
        print(f"  Predicted RUL      : {self.rul_pred:.1f} cycles")
        print(f"  Failure probability: {self.failure_prob:.1%}")
        print(f"  Risk level         : {self.risk_level}")
        if self.recommendation:
            print(f"\n  Recommended action:\n  {self.recommendation}")
        print("=" * 50 + "\n")


def _load_artefacts() -> tuple[xgb.XGBRegressor, xgb.XGBClassifier, list[str]]:
    """Load both models and the feature column order from artefacts/."""
    missing = [
        p for p in [
            settings.artefacts_dir / "xgb_rul.json",
            settings.artefacts_dir / "xgb_failure_cls.json",
            settings.feature_cols_path,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing artefacts: {[str(p) for p in missing]}\n"
            f"Run: python main.py --train"
        )

    reg = xgb.XGBRegressor()
    cls = xgb.XGBClassifier()
    reg.load_model(str(settings.artefacts_dir / "xgb_rul.json"))
    cls.load_model(str(settings.artefacts_dir / "xgb_failure_cls.json"))
    feature_cols = json.loads(settings.feature_cols_path.read_text())

    return reg, cls, feature_cols


def _get_engine_row(engine_id: int, feature_cols: list[str]) -> tuple[np.ndarray, dict]:
    """Fetch the most recent sensor row for an engine from the Parquet file."""
    with get_spark() as spark:
        from pyspark.sql import functions as F
        df     = spark.read.parquet(str(settings.parquet_path))
        latest = (
            df.filter(df.engine_id == engine_id)
              .orderBy(F.col("cycle").desc())
              .limit(1)
              .toPandas()
        )

    if latest.empty:
        raise ValueError(
            f"Engine {engine_id} not found. "
            f"Check the engine ID exists in the dataset."
        )

    X = latest[feature_cols].values

    raw_sensors = [
        c for c in feature_cols
        if c.startswith("sensor_") and "roll" not in c and "lag" not in c
    ]
    snapshot = {col: float(latest[col].values[0]) for col in raw_sensors}

    return X, snapshot


def _rag_diagnose(
    engine_id: int,
    rul_pred: float,
    failure_prob: float,
    sensor_snapshot: dict,
) -> str | None:
    """
    Query the maintenance manual PDF for repair guidance.
    Returns None if no manual is available rather than crashing.
    """
    manual_pdfs = list(settings.manuals_dir.glob("*.pdf")) if settings.manuals_dir.exists() else []
    if not manual_pdfs:
        print(f"  [RAG] No PDFs found in {settings.manuals_dir}/ - skipping.")
        return None

    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.anthropic import Anthropic as AnthropicLLM

        Settings.embed_model = HuggingFaceEmbedding(model_name=settings.rag.embed_model)
        Settings.llm         = AnthropicLLM(model=settings.rag.llm_model)

        print("  Building RAG index...")
        index    = VectorStoreIndex.from_documents(
            SimpleDirectoryReader(str(settings.manuals_dir)).load_data()
        )
        sensor_str = ", ".join(f"{k}={v:.2f}" for k, v in sensor_snapshot.items())
        prompt = f"""
        Engine #{engine_id} anomaly:
        - Predicted RUL: {rul_pred:.1f} cycles
        - Failure probability: {failure_prob:.1%}
        - Sensor readings: {sensor_str}

        Based on the maintenance manual, what is the most likely cause
        and recommended corrective action? Be specific, under 80 words.
        """
        return str(index.as_query_engine().query(prompt))

    except ImportError:
        print("  [RAG] llama-index not installed — skipping.")
        return None


def run_predict(engine_id: int, use_rag: bool = True) -> RiskReport:
    """Full prediction for one engine."""
    reg, cls, feature_cols = _load_artefacts()
    X, snapshot            = _get_engine_row(engine_id, feature_cols)

    rul_pred     = float(reg.predict(X)[0])
    failure_prob = float(cls.predict_proba(X)[0][1])
    risk_level   = "HIGH" if failure_prob > 0.8 else "MEDIUM" if failure_prob > 0.4 else "LOW"

    recommendation = None
    if use_rag and failure_prob >= settings.rag.trigger_threshold:
        recommendation = _rag_diagnose(engine_id, rul_pred, failure_prob, snapshot)

    return RiskReport(
        engine_id=engine_id,
        rul_pred=rul_pred,
        failure_prob=failure_prob,
        risk_level=risk_level,
        sensor_snapshot=snapshot,
        recommendation=recommendation,
    )