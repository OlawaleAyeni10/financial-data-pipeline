"""
test_gold_functional.py
-----------------------
Functional tests for the Gold layer.

Tests the full Gold layer pipeline end to end —
verifying business aggregations produce correct
and complete output from the Silver layer.
"""

import pytest
import os
from pyspark.sql import functions as F
from src.bronze_layer import run_bronze_layer
from src.silver_layer import run_silver_layer
from src.gold_layer import run_gold_layer


@pytest.fixture(scope="module")
def gold_tables(spark):
    """Run full pipeline and return Gold tables dictionary."""
    df_bronze = run_bronze_layer(spark)
    df_silver = run_silver_layer(spark, df_bronze)
    return run_gold_layer(spark, df_silver)


class TestCustomerSummaryFunctional:

    def test_customer_summary_has_correct_record_count(self, gold_tables):
        """Verify customer summary has expected number of customers."""
        assert gold_tables["customer_summary"].count() == 200

    def test_customer_summary_no_duplicate_customers(self, gold_tables):
        """Verify one row per customer in customer summary."""
        df = gold_tables["customer_summary"]
        total = df.count()
        distinct = df.dropDuplicates(["customer_id"]).count()
        assert total == distinct

    def test_customer_summary_segments_are_valid(self, gold_tables):
        """Verify all customer segments are valid values."""
        valid_segments = ["Premium", "High Value", "Mid Tier", "Standard"]
        df = gold_tables["customer_summary"]
        segments = [
            row["customer_segment"]
            for row in df.select("customer_segment").distinct().collect()
        ]
        for segment in segments:
            assert segment in valid_segments

    def test_customer_summary_fraud_score_between_0_and_100(
        self, gold_tables
    ):
        """Verify fraud_risk_score is between 0 and 100."""
        df = gold_tables["customer_summary"]
        invalid = df.filter(
            (F.col("fraud_risk_score") < 0) |
            (F.col("fraud_risk_score") > 100)
        ).count()
        assert invalid == 0

    def test_customer_summary_parquet_exists(self):
        """Verify customer_summary Parquet files exist on disk."""
        assert os.path.exists("output/gold/customer_summary")


class TestMerchantAnalyticsFunctional:

    def test_merchant_analytics_has_correct_record_count(self, gold_tables):
        """Verify merchant analytics has expected number of merchants."""
        assert gold_tables["merchant_analytics"].count() == 50

    def test_merchant_analytics_no_null_categories(self, gold_tables):
        """Verify no NULL merchant categories in output."""
        df = gold_tables["merchant_analytics"]
        null_count = df.filter(
            F.col("merchant_category").isNull()
        ).count()
        assert null_count == 0

    def test_merchant_risk_flags_are_valid(self, gold_tables):
        """Verify merchant risk flags are valid values."""
        valid_flags = ["High Risk", "Medium Risk", "Low Risk"]
        df = gold_tables["merchant_analytics"]
        flags = [
            row["merchant_risk_flag"]
            for row in df.select("merchant_risk_flag").distinct().collect()
        ]
        for flag in flags:
            assert flag in valid_flags

    def test_merchant_rank_starts_at_1(self, gold_tables):
        """Verify minimum rank within category is 1."""
        df = gold_tables["merchant_analytics"]
        min_rank = df.agg(
            F.min("rank_in_category")
        ).collect()[0][0]
        assert min_rank == 1

    def test_merchant_analytics_parquet_exists(self):
        """Verify merchant_analytics Parquet files exist on disk."""
        assert os.path.exists("output/gold/merchant_analytics")


class TestMonthlyTrendsFunctional:

    def test_monthly_trends_has_correct_record_count(self, gold_tables):
        """Verify monthly trends has expected number of months."""
        assert gold_tables["monthly_trends"].count() == 13

    def test_monthly_trends_ordered_chronologically(self, gold_tables):
        """Verify months are in ascending chronological order."""
        df = gold_tables["monthly_trends"]
        months = [
            row["transaction_month"]
            for row in df.orderBy("transaction_month").collect()
        ]
        assert months == sorted(months)

    def test_first_month_mom_growth_is_null(self, gold_tables):
        """Verify first month has NULL mom growth."""
        df = gold_tables["monthly_trends"]
        first_month = df.orderBy("transaction_month") \
            .select("mom_volume_growth_pct").collect()[0][0]
        assert first_month is None

    def test_running_total_is_monotonically_increasing(self, gold_tables):
        """Verify running total volume never decreases."""
        df = gold_tables["monthly_trends"]
        totals = [
            row["running_total_volume"]
            for row in df.orderBy("transaction_month").collect()
        ]
        assert all(
            totals[i] <= totals[i + 1]
            for i in range(len(totals) - 1)
        )

    def test_monthly_trends_parquet_exists(self):
        """Verify monthly_trends Parquet files exist on disk."""
        assert os.path.exists("output/gold/monthly_trends")

    def test_total_volume_matches_pipeline_summary(self, gold_tables):
        """Verify total volume across all months matches expected."""
        df = gold_tables["monthly_trends"]
        total = df.agg(F.sum("total_volume")).collect()[0][0]
        assert round(total, 2) == 4699772.95