from pydantic import BaseModel, computed_field
from pathlib import Path
from pydantic_settings import BaseSettings

class SparkConfig(BaseModel):
    master:             str = "local[*]"
    app_name:           str = "RUL_Prediction"
    driver_memory:      str = "4g"
    log_level:          str = "WARN"

class FeatureConfig(BaseModel):
    rolling_window:     int = 5
    sensor_columns: list[str] = [
        "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
        "sensor_8",  "sensor_11", "sensor_12", "sensor_13",
        "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ] # sensors with non-zero variance

class XGBRegressorConfig(BaseModel):
    n_estimators:       int = 300
    max_depth:          int = 6
    learning_rate:      float = 0.6
    subsample:          float = 0.8
    colsample_bytree:   float = 0.8
    random_state:       int = 42
    failure_cycles:     int = 30 # if RUL < this = failure soon

class XGBClassifierConfig(BaseModel):
    n_estimators:       int = 300
    max_depth:          int = 5
    learning_rate:      float = 0.05
    random_state:       int = 42

class RAGConfig(BaseModel):
    embed_model:        str = "BAAI/bge-small-en-v1.5"
    llm_model:          str = "gemini-3-flash-preview"
    trigger_threshold:  float = 0.4

class Settings(BaseSettings):
    data_raw_dir:       Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    artefacts_dir:      Path = Path("artifacts")
    manuals_dir:        Path = Path("manuals")
    logs_dir:           Path = Path("logs")

    spark:              SparkConfig = SparkConfig()
    features:           FeatureConfig = FeatureConfig()
    xgb_regressor:      XGBRegressorConfig = XGBRegressorConfig()
    xgb_classifier:     XGBClassifierConfig = XGBClassifierConfig()
    rag:                RAGConfig = RAGConfig()

    @computed_field
    @property
    def train_raw_path(self) -> Path:
        return self.data_raw_dir / "train_FD001.txt"

    @computed_field
    @property
    def parquet_path(self) -> Path:
        return self.data_processed_dir / "train_features.parquet"

    @computed_field
    @property
    def feature_cols_path(self) -> Path:
        return self.artefacts_dir / "feature_cols.json"

settings = Settings()