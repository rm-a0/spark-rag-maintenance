from contextlib import contextmanager

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import settings

COL_NAMES = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

@contextmanager
def get_spark():
    """
    Context manager so Spark always shuts down cleanly,
    even if the pipeline throws an exception mid-run.
    """
    cfg   = settings.spark
    spark = (
        SparkSession.builder
        .master(cfg.master)
        .appName(cfg.app_name)
        .config("spark.driver.memory", cfg.driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg.log_level)
    try:
        yield spark
    finally:
        spark.stop()


def load_raw(spark: SparkSession, path: str) -> DataFrame:
    """
    Load a C-MAPSS whitespace-separated file.
    Drops the two trailing phantom columns caused by trailing spaces in the file.
    """
    df = spark.read.csv(path, sep=" ", inferSchema=True)
    phantom = [c for c in df.columns if df.filter(F.col(c).isNotNull()).count() == 0]
    df = df.drop(*phantom)

    return df.toDF(*COL_NAMES)


def add_labels(df: DataFrame) -> DataFrame:
    """
    Add RUL and failure_soon columns.

    RUL = max_cycle - current_cycle per engine
    failure_soon = 1 if RUL <= failure_cycles threshold, else 0
    """
    max_cycles = df.groupBy("engine_id").agg(F.max("cycle").alias("max_cycle"))
    df = df.join(max_cycles, on="engine_id")
    df = df.withColumn("RUL", F.col("max_cycle") - F.col("cycle"))
    df = df.withColumn(
        "failure_soon",
        (F.col("RUL") <= settings.xgb_regressor.failure_cycles).cast("int")
    )
    return df


def add_rolling_features(df: DataFrame) -> DataFrame:
    """Add rolling mean, std, and lag-1 per sensor column."""
    window_size = settings.features.rolling_window
    look_back   = -(window_size - 1)

    w_roll = (
        Window
        .partitionBy("engine_id")
        .orderBy("cycle")
        .rowsBetween(look_back, 0)
    )
    w_lag = (
        Window
        .partitionBy("engine_id")
        .orderBy("cycle")
    )

    for col in settings.features.sensor_columns:
        df = (
            df
            .withColumn(f"{col}_roll_mean", F.mean(col).over(w_roll))
            .withColumn(f"{col}_roll_std",  F.stddev(col).over(w_roll))
            .withColumn(f"{col}_lag1",      F.lag(col, 1).over(w_lag))
        )

    # Drop rows where rolling features are null
    return df.dropna()


def run_pipeline() -> None:
    """Full pipeline."""
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    with get_spark() as spark:
        print("Loading raw data...")
        df = load_raw(spark, str(settings.train_raw_path))

        print("Adding labels...")
        df = add_labels(df)

        print("Engineering features...")
        df = add_rolling_features(df)

        print(f"Saving Parquet → {settings.parquet_path}")
        df.write.mode("overwrite").parquet(str(settings.parquet_path))

    print(f"Pipeline complete.")