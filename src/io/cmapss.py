from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pathlib import Path

COL_NAMES = [
    "engine_id",
    "cycle",
    *[f"setting_{i}" for i in range(1, 4) ],
    *[f"sensor_{i}" for i in range(1, 22)]
]

def load_cmapss(spark: SparkSession, path: Path) -> DataFrame:
    df = spark.read.csv(str(path), sep=" ", inferSchema=True)
    # Drop empty columns and rename the rest
    phantom = [c for c in df.columns if df.filter(F.col(c).isNotNull()).count() == 0]
    df = df.drop(*phantom)
    return df.toDF(*COL_NAMES)