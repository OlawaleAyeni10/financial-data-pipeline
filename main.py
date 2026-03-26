"""
main.py
-------
Pipeline Orchestrator - Financial Transactions Data Pipeline

This script orchestrates the full Medallion Architecture pipeline:
    Bronze  → Raw ingestion from landing zone
    Silver  → Cleaning, validation and standardisation
    Gold    → Business aggregations and analytics

Usage:
    python main.py

In a real Azure/Databricks environment this would be triggered
by Azure Data Factory or a Databricks Job scheduler, with
parameters passed in via widgets or job configurations.

Architecture:
    Source System (CSV / API / Database)
        ↓
    Azure Data Lake Storage Gen2 (Landing Zone)
        ↓
    Bronze Layer (Raw Parquet)
        ↓
    Silver Layer (Cleaned Parquet, partitioned by month)
        ↓
    Gold Layer (Business aggregations, Unity Catalog tables)
        ↓
    Power BI / Azure ML / Downstream consumers
"""

from pyspark.sql import SparkSession
from src.bronze_layer import run_bronze_layer
from src.silver_layer import run_silver_layer
from src.gold_layer import run_gold_layer
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """
    Create and configure the SparkSession.

    In a real Databricks environment this would be provided
    automatically by the cluster — no manual creation needed.
    Locally we configure it to use all available CPU cores
    via local[*] for optimal performance.

    Returns:
        Configured SparkSession
    """
    spark = SparkSession.builder \
        .appName("FinancialTransactionsPipeline") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def log_pipeline_summary(gold_tables: dict) -> None:
    """
    Print a summary of the pipeline outputs to the console.

    In production this would publish metrics to Azure Monitor
    or a data observability platform like Monte Carlo.

    Args:
        gold_tables: Dictionary of Gold layer DataFrames
    """
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    customer_summary = gold_tables["customer_summary"]
    merchant_analytics = gold_tables["merchant_analytics"]
    monthly_trends = gold_tables["monthly_trends"]

    logger.info(f"Total customers processed : "
                f"{customer_summary.count()}")
    logger.info(f"Total merchants analysed  : "
                f"{merchant_analytics.count()}")
    logger.info(f"Months of data covered    : "
                f"{monthly_trends.count()}")

    # High risk merchants
    high_risk = merchant_analytics.filter(
        merchant_analytics.merchant_risk_flag == "High Risk"
    ).count()
    logger.info(f"High risk merchants       : {high_risk}")

    # Premium customers
    premium = customer_summary.filter(
        customer_summary.customer_segment == "Premium"
    ).count()
    logger.info(f"Premium customers         : {premium}")

    # Total transaction volume
    total_volume = monthly_trends.agg({"total_volume": "sum"}) \
        .collect()[0][0]
    logger.info(f"Total transaction volume  : £{total_volume:,.2f}")

    logger.info("=" * 60)

    print("\n=== Sample: Customer Summary ===")
    customer_summary.select(
        "customer_id",
        "total_transactions",
        "total_spend",
        "customer_segment",
        "fraud_risk_score",
        "preferred_channel",
        "preferred_category"
    ).show(5, truncate=False)

    print("\n=== Sample: Merchant Analytics ===")
    merchant_analytics.select(
        "merchant_name",
        "merchant_category",
        "total_transactions",
        "total_revenue",
        "unique_customers",
        "merchant_risk_flag",
        "rank_in_category"
    ).show(5, truncate=False)

    print("\n=== Sample: Monthly Trends ===")
    monthly_trends.select(
        "transaction_month",
        "total_transactions",
        "total_volume",
        "active_customers",
        "mom_volume_growth_pct",
        "running_total_volume"
    ).show(12, truncate=False)


def main():
    """
    Main pipeline entry point.
    Runs the full Bronze → Silver → Gold pipeline.
    """
    logger.info("Starting Financial Transactions Data Pipeline")
    logger.info("Architecture: Medallion (Bronze → Silver → Gold)")
    pipeline_start = time.time()

    spark = create_spark_session()

    try:
        # Bronze layer
        logger.info("Phase 1/3: Bronze Layer")
        bronze_start = time.time()
        df_bronze = run_bronze_layer(spark)
        logger.info(f"Bronze complete in "
                    f"{round(time.time() - bronze_start, 2)}s")

        # Silver layer
        logger.info("Phase 2/3: Silver Layer")
        silver_start = time.time()
        df_silver = run_silver_layer(spark, df_bronze)
        logger.info(f"Silver complete in "
                    f"{round(time.time() - silver_start, 2)}s")

        # Gold layer
        logger.info("Phase 3/3: Gold Layer")
        gold_start = time.time()
        gold_tables = run_gold_layer(spark, df_silver)
        logger.info(f"Gold complete in "
                    f"{round(time.time() - gold_start, 2)}s")

        # Pipeline summary
        log_pipeline_summary(gold_tables)

        total_time = round(time.time() - pipeline_start, 2)
        logger.info(f"Pipeline completed successfully in {total_time}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

    finally:
        spark.stop()
        logger.info("SparkSession closed.")


if __name__ == "__main__":
    main()