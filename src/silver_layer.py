"""
silver_layer.py
---------------
Silver Layer - Data Cleaning & Validation

Responsibilities:
- Remove duplicate transactions
- Drop records with nulls in critical fields
- Validate and cast data types
- Standardise categorical fields (currency, channel, status)
- Remove invalid amounts (negative or zero values)
- Combine transaction_date and transaction_time into a timestamp
- Add data quality metadata columns
- Write cleaned data to Parquet (output/silver/)

In a real Azure/Databricks environment this layer would also
integrate with Unity Catalog for data lineage tracking and
Azure Purview for governance and compliance — particularly
important in regulated industries like financial services.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SILVER_OUTPUT_PATH = "output/silver/transactions"

VALID_CURRENCIES = ["GBP", "USD", "EUR"]
VALID_CHANNELS = ["online", "in-store", "mobile", "ATM"]
VALID_STATUSES = ["completed", "pending", "failed", "reversed"]
CRITICAL_FIELDS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "amount",
    "transaction_date"
]


def remove_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove duplicate transactions based on transaction_id.

    In financial services, duplicate transactions can arise from
    double-processing errors, retries, or system failures.
    We keep the first occurrence and drop subsequent duplicates.

    Args:
        df: Bronze DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    before = df.count()
    df = df.dropDuplicates(["transaction_id"])
    after = df.count()
    logger.info(f"Duplicates removed: {before - after}")
    return df


def drop_critical_nulls(df: DataFrame) -> DataFrame:
    """
    Drop records where critical fields are null.

    Critical fields are those without which a transaction record
    is meaningless for downstream analytics. Non-critical nulls
    (e.g. customer_location) are retained and handled downstream.

    Args:
        df: DataFrame after deduplication

    Returns:
        DataFrame with null critical records removed
    """
    before = df.count()
    df = df.dropna(subset=CRITICAL_FIELDS)
    after = df.count()
    logger.info(f"Records dropped due to critical nulls: {before - after}")
    return df


def validate_amounts(df: DataFrame) -> DataFrame:
    """
    Remove records with invalid transaction amounts.

    Negative or zero amounts indicate data entry errors or
    failed reversals that were not properly handled upstream.
    These are removed to prevent skewing financial aggregations.

    Args:
        df: DataFrame after null handling

    Returns:
        DataFrame with only valid positive amounts
    """
    before = df.count()
    df = df.filter(F.col("amount") > 0)
    after = df.count()
    logger.info(f"Records dropped due to invalid amounts: {before - after}")
    return df


def standardise_categoricals(df: DataFrame) -> DataFrame:
    """
    Standardise categorical fields to ensure consistency.

    Replaces unexpected values with 'unknown' rather than dropping
    the record — preserving as much data as possible while flagging
    anomalies for downstream consumers to handle as appropriate.

    Args:
        df: DataFrame after amount validation

    Returns:
        DataFrame with standardised categorical values
    """
    df = df.withColumn(
        "currency",
        F.when(F.col("currency").isin(VALID_CURRENCIES), F.col("currency"))
        .otherwise(F.lit("unknown"))
    )

    df = df.withColumn(
        "channel",
        F.when(F.col("channel").isin(VALID_CHANNELS), F.col("channel"))
        .otherwise(F.lit("unknown"))
    )

    df = df.withColumn(
        "status",
        F.when(F.col("status").isin(VALID_STATUSES), F.col("status"))
        .otherwise(F.lit("unknown"))
    )

    return df


def build_transaction_timestamp(df: DataFrame) -> DataFrame:
    """
    Combine transaction_date and transaction_time into a single
    transaction_timestamp column of TimestampType.

    Having a single timestamp field simplifies time-based
    analytics and window functions in the Gold layer.

    Args:
        df: DataFrame after categorical standardisation

    Returns:
        DataFrame with transaction_timestamp column added
    """
    df = df.withColumn(
        "transaction_timestamp",
        F.to_timestamp(
            F.concat_ws(" ", F.col("transaction_date"), F.col("transaction_time")),
            "yyyy-MM-dd HH:mm:ss"
        ).cast(TimestampType())
    )

    df = df.withColumn(
        "transaction_month",
        F.date_format(F.col("transaction_timestamp"), "yyyy-MM")
    )

    df = df.withColumn(
        "day_of_week",
        F.dayofweek(F.col("transaction_timestamp"))
    )

    return df


def add_silver_metadata(df: DataFrame) -> DataFrame:
    """
    Add Silver layer metadata columns for auditability.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with metadata columns added
    """
    return df \
        .withColumn("silver_processed_timestamp", F.current_timestamp()) \
        .withColumn("pipeline_layer", F.lit("silver"))


def write_silver(df: DataFrame) -> None:
    """
    Write the Silver layer output to Parquet format,
    partitioned by transaction_month for query efficiency.

    Partitioning by month is a common pattern in financial
    services pipelines, most analytical queries filter
    by time period, so partitioning eliminates unnecessary
    file scans and significantly improves query performance.

    Args:
        df: Fully cleaned and validated DataFrame
    """
    logger.info(f"Writing Silver layer to: {SILVER_OUTPUT_PATH}")

    df.write \
        .mode("overwrite") \
        .partitionBy("transaction_month") \
        .parquet(SILVER_OUTPUT_PATH)

    logger.info("Silver layer write complete.")


def run_silver_layer(spark: SparkSession, df_bronze: DataFrame) -> DataFrame:
    """
    Orchestrates the full Silver layer pipeline.

    Args:
        spark: Active SparkSession
        df_bronze: Bronze layer DataFrame

    Returns:
        Silver DataFrame (for passing downstream to Gold)
    """
    logger.info("Starting Silver layer...")

    df = remove_duplicates(df_bronze)
    df = drop_critical_nulls(df)
    df = validate_amounts(df)
    df = standardise_categoricals(df)
    df = build_transaction_timestamp(df)
    df = add_silver_metadata(df)

    write_silver(df)

    logger.info(f"Silver layer complete. Records: {df.count()}")
    return df