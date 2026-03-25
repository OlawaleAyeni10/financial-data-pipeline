"""
bronze_layer.py
---------------
Bronze Layer - Raw Data Ingestion

Responsibilities:
- Read raw CSV data from the landing zone (data/raw/)
- Enforce an explicit schema to ensure consistent data types
- Add ingestion metadata (load timestamp, source file)
- Write raw data as-is to Parquet format (output/bronze/)

No business logic or cleaning is applied at this stage.
The Bronze layer preserves the raw data exactly as received,
providing a reliable reprocessing baseline if downstream
logic needs to change.

In a real Azure environment this would read from
Azure Data Lake Storage Gen2 and write back to ADLS Gen2,
with the Parquet files partitioned by ingestion date.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType
)
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_INPUT_PATH = "data/raw/transactions.csv"
BRONZE_OUTPUT_PATH = "output/bronze/transactions"

# ── Schema Definition ────────────────────────────────────────────────────────
# Explicitly defining the schema rather than inferring it is best practice
# in production pipelines — it's faster, safer, and catches upstream
# schema changes early.
RAW_SCHEMA = StructType([
    StructField("transaction_id",    StringType(),  True),
    StructField("customer_id",       StringType(),  True),
    StructField("merchant_id",       StringType(),  True),
    StructField("merchant_name",     StringType(),  True),
    StructField("merchant_category", StringType(),  True),
    StructField("amount",            DoubleType(),  True),
    StructField("currency",          StringType(),  True),
    StructField("channel",           StringType(),  True),
    StructField("status",            StringType(),  True),
    StructField("transaction_date",  StringType(),  True),
    StructField("transaction_time",  StringType(),  True),
    StructField("customer_location", StringType(),  True),
    StructField("is_flagged",        BooleanType(), True),
])


def read_raw_transactions(spark: SparkSession) -> "DataFrame":
    """
    Read raw transaction CSV from the landing zone.

    Args:
        spark: Active SparkSession

    Returns:
        Raw DataFrame with enforced schema
    """
    logger.info(f"Reading raw transactions from: {RAW_INPUT_PATH}")

    df = spark.read \
        .option("header", "true") \
        .option("nullValue", "") \
        .option("mode", "PERMISSIVE") \
        .schema(RAW_SCHEMA) \
        .csv(RAW_INPUT_PATH)

    logger.info(f"Raw records ingested: {df.count()}")
    return df


def add_ingestion_metadata(df: "DataFrame") -> "DataFrame":
    """
    Add metadata columns to track when and where data was ingested.
    This is standard practice in production pipelines for auditability
    and reprocessing — critical in regulated industries like financial
    services where data lineage must be traceable.

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with added metadata columns
    """
    return df \
        .withColumn("ingestion_timestamp", F.current_timestamp()) \
        .withColumn("source_file", F.lit(RAW_INPUT_PATH)) \
        .withColumn("pipeline_layer", F.lit("bronze"))


def write_bronze(df: "DataFrame") -> None:
    """
    Write the Bronze layer output to Parquet format.

    Parquet is the standard format for Azure Data Lake Storage Gen2
    — it is columnar, compressed, and optimised for analytical queries
    in Databricks and Spark.

    In production this would be partitioned by ingestion date:
    .partitionBy("ingestion_date")

    Args:
        df: DataFrame with ingestion metadata added
    """
    logger.info(f"Writing Bronze layer to: {BRONZE_OUTPUT_PATH}")

    df.write \
        .mode("overwrite") \
        .parquet(BRONZE_OUTPUT_PATH)

    logger.info("Bronze layer write complete.")


def run_bronze_layer(spark: SparkSession) -> "DataFrame":
    """
    Orchestrates the full Bronze layer pipeline.

    Args:
        spark: Active SparkSession

    Returns:
        Bronze DataFrame (for passing downstream to Silver)
    """
    logger.info("Starting Bronze layer...")

    df_raw = read_raw_transactions(spark)
    df_bronze = add_ingestion_metadata(df_raw)
    write_bronze(df_bronze)

    logger.info(f"Bronze layer complete. Records: {df_bronze.count()}")
    return df_bronze