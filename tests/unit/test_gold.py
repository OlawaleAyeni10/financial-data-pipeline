"""
test_gold.py
------------
Unit tests for the Gold layer.

Tests aggregation logic and business rules applied
in the Gold layer using small known datasets.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType, TimestampType
)
from pyspark.sql import functions as F
from src.gold_layer import (
    build_customer_summary,
    build_merchant_analytics,
    build_monthly_trends
)


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("GoldUnitTests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_silver_df(spark):
    """
    Create a small Silver-style DataFrame with known values
    so we can assert exact aggregation results.
    """
    schema = StructType([
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
        StructField("transaction_timestamp", StringType(), True),
        StructField("transaction_month", StringType(),  True),
        StructField("day_of_week",       StringType(),  True),
    ])

    data = [
        ("TXN-001", "CUST-001", "MERCH-001", "Merchant A",
         "Groceries", 100.0, "GBP", "online", "completed",
         "2025-06-01", "10:00:00", "London", False,
         "2025-06-01 10:00:00", "2025-06", "1"),
        ("TXN-002", "CUST-001", "MERCH-001", "Merchant A",
         "Groceries", 200.0, "GBP", "online", "completed",
         "2025-06-02", "11:00:00", "London", False,
         "2025-06-02 11:00:00", "2025-06", "2"),
        ("TXN-003", "CUST-002", "MERCH-002", "Merchant B",
         "Transport", 50.0, "GBP", "mobile", "completed",
         "2025-06-03", "12:00:00", "Manchester", False,
         "2025-06-03 12:00:00", "2025-06", "3"),
        ("TXN-004", "CUST-002", "MERCH-002", "Merchant B",
         "Transport", 75000.0, "GBP", "mobile", "completed",
         "2025-07-01", "09:00:00", "Manchester", True,
         "2025-07-01 09:00:00", "2025-07", "2"),
        ("TXN-005", "CUST-003", "MERCH-001", "Merchant A",
         "Groceries", 150.0, "GBP", "in-store", "failed",
         "2025-07-02", "15:00:00", "Leeds", False,
         "2025-07-02 15:00:00", "2025-07", "3"),
    ]
    return spark.createDataFrame(data, schema=schema)


class TestBuildCustomerSummary:

    def test_one_row_per_customer(self, sample_silver_df):
        """Verify output has exactly one row per customer."""
        df = build_customer_summary(sample_silver_df)
        total_rows = df.count()
        distinct_customers = sample_silver_df \
            .select("customer_id").distinct().count()
        assert total_rows == distinct_customers

    def test_correct_total_spend_per_customer(self, sample_silver_df):
        """Verify total_spend is correctly summed per customer."""
        df = build_customer_summary(sample_silver_df)
        cust1 = df.filter(F.col("customer_id") == "CUST-001") \
            .select("total_spend").collect()[0][0]
        assert cust1 == 300.0

    def test_customer_segment_column_exists(self, sample_silver_df):
        """Verify customer_segment column is present."""
        df = build_customer_summary(sample_silver_df)
        assert "customer_segment" in df.columns

    def test_fraud_risk_score_column_exists(self, sample_silver_df):
        """Verify fraud_risk_score column is present."""
        df = build_customer_summary(sample_silver_df)
        assert "fraud_risk_score" in df.columns

    def test_fraud_risk_score_is_non_negative(self, sample_silver_df):
        """Verify fraud_risk_score is never negative."""
        df = build_customer_summary(sample_silver_df)
        scores = [row["fraud_risk_score"] for row in df.collect()]
        assert all(s >= 0 for s in scores)

    def test_premium_segment_assigned_correctly(self, sample_silver_df):
        """Verify customers with spend >= 50000 are Premium."""
        df = build_customer_summary(sample_silver_df)
        cust2 = df.filter(F.col("customer_id") == "CUST-002") \
            .select("customer_segment").collect()[0][0]
        assert cust2 == "Premium"


class TestBuildMerchantAnalytics:

    def test_one_row_per_merchant(self, sample_silver_df):
        """Verify output has one row per merchant."""
        df = build_merchant_analytics(sample_silver_df)
        total_rows = df.count()
        distinct_merchants = sample_silver_df \
            .filter(F.col("merchant_category").isNotNull()) \
            .select("merchant_id").distinct().count()
        assert total_rows == distinct_merchants

    def test_merchant_risk_flag_column_exists(self, sample_silver_df):
        """Verify merchant_risk_flag column is present."""
        df = build_merchant_analytics(sample_silver_df)
        assert "merchant_risk_flag" in df.columns

    def test_rank_in_category_column_exists(self, sample_silver_df):
        """Verify rank_in_category column is present."""
        df = build_merchant_analytics(sample_silver_df)
        assert "rank_in_category" in df.columns

    def test_no_null_merchant_categories(self, sample_silver_df):
        """Verify no NULL merchant categories in output."""
        df = build_merchant_analytics(sample_silver_df)
        null_count = df.filter(
            F.col("merchant_category").isNull()
        ).count()
        assert null_count == 0


class TestBuildMonthlyTrends:

    def test_one_row_per_month(self, sample_silver_df):
        """Verify output has one row per transaction_month."""
        df = build_monthly_trends(sample_silver_df)
        total_rows = df.count()
        distinct_months = sample_silver_df \
            .select("transaction_month").distinct().count()
        assert total_rows == distinct_months

    def test_mom_growth_column_exists(self, sample_silver_df):
        """Verify mom_volume_growth_pct column is present."""
        df = build_monthly_trends(sample_silver_df)
        assert "mom_volume_growth_pct" in df.columns

    def test_running_total_column_exists(self, sample_silver_df):
        """Verify running_total_volume column is present."""
        df = build_monthly_trends(sample_silver_df)
        assert "running_total_volume" in df.columns

    def test_first_month_has_null_mom_growth(self, sample_silver_df):
        """Verify first month has NULL mom growth (no prior month)."""
        df = build_monthly_trends(sample_silver_df)
        first_month = df.orderBy("transaction_month") \
            .select("mom_volume_growth_pct").collect()[0][0]
        assert first_month is None

    def test_running_total_is_cumulative(self, sample_silver_df):
        """Verify running total increases monotonically."""
        df = build_monthly_trends(sample_silver_df)
        totals = [
            row["running_total_volume"]
            for row in df.orderBy("transaction_month").collect()
        ]
        assert all(
            totals[i] <= totals[i + 1]
            for i in range(len(totals) - 1)
        )