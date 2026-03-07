import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession


def save_parquet(df: pd.DataFrame | SparkDataFrame, path: Path) -> None:
    """Save a Pandas or Spark DataFrame as Parquet."""
    if isinstance(df, pd.DataFrame):
        df.to_parquet(path, index=False)
    elif isinstance(df, SparkDataFrame):
        df.write.mode("overwrite").parquet(str(path))
    else:
        raise TypeError(f"Expected pandas or Spark DataFrame, got {type(df)}")


def load_parquet_pandas(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_parquet_spark(spark: SparkSession, path: Path) -> SparkDataFrame:
    return spark.read.parquet(str(path))