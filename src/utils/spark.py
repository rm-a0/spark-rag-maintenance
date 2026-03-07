from contextlib import contextmanager
from pyspark.sql import SparkSession
from src.config import SparkConfig, settings

@contextmanager
def spark_session(cfg: SparkConfig):
    """Context manager for SparkSession."""
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
