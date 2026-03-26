"""
gold_layer.py
-------------
Gold Layer - Business Aggregations & Analytics

Responsibilities:
- Customer spending summaries and behaviour analysis
- Merchant performance analytics
- Fraud risk scoring using window functions
- Monthly transaction trends
- Write business-ready aggregations to Parquet (output/gold/)

The Gold layer produces business-ready datasets optimised
for consumption by BI tools, dashboards, and ML models.
In a real Azure environment these tables would be registered
in Unity Catalog and served via Power BI or Azure ML.

Key PySpark concepts demonstrated:
- Window functions (ranking, running totals, lag/lead)
- Multi-level aggregations
- Conditional logic with F.when()
- Joining aggregated DataFrames
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLD_OUTPUT_PATH = "output/gold"


def build_customer_summary(df: DataFrame) -> DataFrame:
    """
    Build a customer-level spending summary.

    Aggregates transaction data to produce a single row per
    customer capturing their overall spending behaviour.
    This is the foundation for customer segmentation and
    personalisation models in financial services.

    Metrics produced:
    - Total and average spend
    - Transaction count and frequency
    - Most used channel and currency
    - Preferred merchant category
    - High value transaction count

    Args:
        df: Silver layer DataFrame

    Returns:
        Customer summary DataFrame
    """
    logger.info("Building customer summary...")

    # Window spec for ranking within customer
    customer_window = Window.partitionBy("customer_id")

    # Most frequent channel per customer
    channel_counts = df.groupBy("customer_id", "channel") \
        .agg(F.count("*").alias("channel_count"))

    channel_window = Window.partitionBy("customer_id") \
        .orderBy(F.col("channel_count").desc())

    preferred_channel = channel_counts \
    .withColumn("rank", F.rank().over(channel_window)) \
    .filter(F.col("rank") == 1) \
    .groupBy("customer_id") \
    .agg(F.first("channel").alias("preferred_channel"))

    # Most frequent merchant category per customer
    category_counts = df.groupBy("customer_id", "merchant_category") \
        .agg(F.count("*").alias("category_count"))

    category_window = Window.partitionBy("customer_id") \
        .orderBy(F.col("category_count").desc())

    preferred_category = category_counts \
    .withColumn("rank", F.rank().over(category_window)) \
    .filter(F.col("rank") == 1) \
    .groupBy("customer_id") \
    .agg(F.first("merchant_category").alias("preferred_category"))

    # Core customer aggregations
    customer_agg = df.groupBy("customer_id").agg(
        F.count("transaction_id").alias("total_transactions"),
        F.round(F.sum("amount"), 2).alias("total_spend"),
        F.round(F.avg("amount"), 2).alias("avg_transaction_value"),
        F.round(F.max("amount"), 2).alias("max_transaction_value"),
        F.round(F.min("amount"), 2).alias("min_transaction_value"),
        F.countDistinct("merchant_id").alias("unique_merchants_used"),
        F.countDistinct("transaction_month").alias("active_months"),
        F.sum(F.when(F.col("is_flagged") == True, 1).otherwise(0))
         .alias("flagged_transaction_count"),
        F.sum(F.when(F.col("status") == "completed", 1).otherwise(0))
         .alias("completed_transactions"),
        F.sum(F.when(F.col("status") == "failed", 1).otherwise(0))
         .alias("failed_transactions"),
        F.max("transaction_timestamp").alias("last_transaction_date"),
        F.min("transaction_timestamp").alias("first_transaction_date")
    )

    # Join preferred channel and category
    customer_summary = customer_agg \
        .join(preferred_channel, on="customer_id", how="left") \
        .join(preferred_category, on="customer_id", how="left")

    # Add customer risk score based on flagged transaction ratio
    customer_summary = customer_summary.withColumn(
        "fraud_risk_score",
        F.round(
            F.col("flagged_transaction_count") / F.col("total_transactions") * 100,
            2
        )
    )

    # Add customer segment based on total spend
    customer_summary = customer_summary.withColumn(
        "customer_segment",
        F.when(F.col("total_spend") >= 50000, "Premium")
        .when(F.col("total_spend") >= 20000, "High Value")
        .when(F.col("total_spend") >= 5000, "Mid Tier")
        .otherwise("Standard")
    )

    logger.info(f"Customer summary built. Records: {customer_summary.count()}")
    return customer_summary


def build_merchant_analytics(df: DataFrame) -> DataFrame:
    """
    Build merchant-level performance analytics.

    Produces a summary of transaction volume and value per
    merchant, including ranking within merchant category.
    This type of analysis is used in financial services for
    merchant risk profiling and partnership analytics.

    Args:
        df: Silver layer DataFrame

    Returns:
        Merchant analytics DataFrame
    """
    logger.info("Building merchant analytics...")

    # Core merchant aggregations
    merchant_agg = df.filter(F.col("merchant_category").isNotNull()) \
    .groupBy(
        "merchant_id",
        "merchant_name",
        "merchant_category"
    ).agg(
        F.count("transaction_id").alias("total_transactions"),
        F.round(F.sum("amount"), 2).alias("total_revenue"),
        F.round(F.avg("amount"), 2).alias("avg_transaction_value"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.sum(F.when(F.col("is_flagged") == True, 1).otherwise(0))
         .alias("flagged_transactions"),
        F.sum(F.when(F.col("status") == "completed", 1).otherwise(0))
         .alias("completed_transactions"),
        F.sum(F.when(F.col("currency") == "GBP", 1).otherwise(0))
         .alias("gbp_transactions"),
        F.sum(F.when(F.col("channel") == "online", 1).otherwise(0))
         .alias("online_transactions")
    )

    # Rank merchants within their category by total revenue
    category_window = Window.partitionBy("merchant_category") \
        .orderBy(F.col("total_revenue").desc())

    merchant_agg = merchant_agg \
        .withColumn("rank_in_category", F.rank().over(category_window))

    # Add merchant risk flag
    merchant_agg = merchant_agg.withColumn(
        "merchant_risk_flag",
        F.when(
            F.col("flagged_transactions") / F.col("total_transactions") > 0.05,
            "High Risk"
        )
        .when(
            F.col("flagged_transactions") / F.col("total_transactions") > 0.02,
            "Medium Risk"
        )
        .otherwise("Low Risk")
    )

    logger.info(f"Merchant analytics built. Records: {merchant_agg.count()}")
    return merchant_agg


def build_monthly_trends(df: DataFrame) -> DataFrame:
    """
    Build monthly transaction trend analysis.

    Uses window functions to calculate month-over-month
    growth rates — a key metric in financial services
    for spotting emerging patterns and anomalies.

    Args:
        df: Silver layer DataFrame

    Returns:
        Monthly trends DataFrame with MoM growth rates
    """
    logger.info("Building monthly trends...")

    # Monthly aggregations
    monthly_agg = df.groupBy("transaction_month").agg(
        F.count("transaction_id").alias("total_transactions"),
        F.round(F.sum("amount"), 2).alias("total_volume"),
        F.round(F.avg("amount"), 2).alias("avg_transaction_value"),
        F.countDistinct("customer_id").alias("active_customers"),
        F.countDistinct("merchant_id").alias("active_merchants"),
        F.sum(F.when(F.col("is_flagged") == True, 1).otherwise(0))
         .alias("flagged_transactions"),
        F.sum(F.when(F.col("status") == "failed", 1).otherwise(0))
         .alias("failed_transactions")
    )

    # Window for month-over-month calculations
    month_window = Window.orderBy("transaction_month")

    # Add month-over-month volume growth
    monthly_agg = monthly_agg \
        .withColumn(
            "prev_month_volume",
            F.lag("total_volume", 1).over(month_window)
        ) \
        .withColumn(
            "mom_volume_growth_pct",
            F.round(
                (F.col("total_volume") - F.col("prev_month_volume"))
                / F.col("prev_month_volume") * 100,
                2
            )
        ) \
        .withColumn(
            "prev_month_transactions",
            F.lag("total_transactions", 1).over(month_window)
        ) \
        .withColumn(
            "mom_transaction_growth_pct",
            F.round(
                (F.col("total_transactions") - F.col("prev_month_transactions"))
                / F.col("prev_month_transactions") * 100,
                2
            )
        ) \
        .withColumn(
            "running_total_volume",
            F.round(F.sum("total_volume").over(
                month_window.rowsBetween(Window.unboundedPreceding, 0)
            ), 2)
        )

    monthly_agg = monthly_agg.orderBy("transaction_month")

    logger.info(f"Monthly trends built. Records: {monthly_agg.count()}")
    return monthly_agg


def write_gold(df: DataFrame, table_name: str) -> None:
    """
    Write a Gold layer table to Parquet.

    Each Gold table is written to its own subfolder,
    mirroring how tables would be registered separately
    in Databricks Unity Catalog for downstream consumption
    by Power BI, Azure ML, or other analytics tools.

    Args:
        df: Aggregated DataFrame to write
        table_name: Name of the Gold table
    """
    output_path = f"{GOLD_OUTPUT_PATH}/{table_name}"
    logger.info(f"Writing Gold table '{table_name}' to: {output_path}")

    df.write \
        .mode("overwrite") \
        .parquet(output_path)

    logger.info(f"Gold table '{table_name}' write complete.")


def run_gold_layer(spark: SparkSession, df_silver: DataFrame) -> dict:
    """
    Orchestrates the full Gold layer pipeline.

    Args:
        spark: Active SparkSession
        df_silver: Silver layer DataFrame

    Returns:
        Dictionary of Gold DataFrames keyed by table name
    """
    logger.info("Starting Gold layer...")

    df_customer_summary = build_customer_summary(df_silver)
    write_gold(df_customer_summary, "customer_summary")

    df_merchant_analytics = build_merchant_analytics(df_silver)
    write_gold(df_merchant_analytics, "merchant_analytics")

    df_monthly_trends = build_monthly_trends(df_silver)
    write_gold(df_monthly_trends, "monthly_trends")

    logger.info("Gold layer complete.")

    return {
        "customer_summary": df_customer_summary,
        "merchant_analytics": df_merchant_analytics,
        "monthly_trends": df_monthly_trends
    }