"""
test_silver_functional.py
-------------------------
Functional tests for the Silver layer.

Tests the full Silver layer pipeline end to end —
verifying cleaning, validation and standardisation
produce the expected output from the Bronze layer.
"""

import pytest
import os
from pyspark.sql import functions as F
from src.bronze_layer import run_bronze_layer
from src.silver_layer import (
    run_silver_layer,
    VALID_CURRENCIES,
    VALID_CHANNELS,
    VALID_STATUSES
)

@pytest.fixture(scope="module")
def silver_df(spark):
    """Run Bronze and Silver layers and return Silver DataFrame."""
    df_bronze = run_bronze_layer(spark)
    return run_silver_layer(spark, df_bronze)


class TestSilverLayerFunctional:

    def test_silver_record_count_less_than_bronze(self, silver_df):
        """Verify Silver has fewer records than Bronze after cleaning."""
        assert silver_df.count() == 4875

    def test_silver_removes_all_duplicates(self, silver_df):
        """Verify no duplicate transaction_ids remain in Silver."""
        total = silver_df.count()
        distinct = silver_df.dropDuplicates(["transaction_id"]).count()
        assert total == distinct

    def test_silver_no_null_critical_fields(self, silver_df):
        """Verify no nulls in critical fields after Silver cleaning."""
        from src.silver_layer import CRITICAL_FIELDS
        for field in CRITICAL_FIELDS:
            null_count = silver_df.filter(
                F.col(field).isNull()
            ).count()
            assert null_count == 0, \
                f"Found nulls in critical field: {field}"

    def test_silver_no_negative_amounts(self, silver_df):
        """Verify all amounts are positive after Silver validation."""
        negative_count = silver_df.filter(
            F.col("amount") <= 0
        ).count()
        assert negative_count == 0

    def test_silver_currencies_are_valid(self, silver_df):
        """Verify all currency values are valid or 'unknown'."""
        currencies = [
            row["currency"]
            for row in silver_df.select("currency").distinct().collect()
        ]
        for currency in currencies:
            assert currency in VALID_CURRENCIES + ["unknown"]

    def test_silver_channels_are_valid(self, silver_df):
        """Verify all channel values are valid or 'unknown'."""
        channels = [
            row["channel"]
            for row in silver_df.select("channel").distinct().collect()
        ]
        for channel in channels:
            assert channel in VALID_CHANNELS + ["unknown"]

    def test_silver_statuses_are_valid(self, silver_df):
        """Verify all status values are valid or 'unknown'."""
        statuses = [
            row["status"]
            for row in silver_df.select("status").distinct().collect()
        ]
        for status in statuses:
            assert status in VALID_STATUSES + ["unknown"]

    def test_silver_has_transaction_timestamp(self, silver_df):
        """Verify transaction_timestamp column is present and non-null."""
        assert "transaction_timestamp" in silver_df.columns
        null_count = silver_df.filter(
            F.col("transaction_timestamp").isNull()
        ).count()
        assert null_count == 0

    def test_silver_has_transaction_month(self, silver_df):
        """Verify transaction_month column is present."""
        assert "transaction_month" in silver_df.columns

    def test_silver_parquet_partitioned_by_month(self):
        """Verify Silver output is partitioned by transaction_month."""
        assert os.path.exists("output/silver/transactions")
        partitions = [
            f for f in os.listdir("output/silver/transactions")
            if f.startswith("transaction_month=")
        ]
        assert len(partitions) > 0

    def test_silver_pipeline_layer_is_silver(self, silver_df):
        """Verify pipeline_layer is set to silver for all records."""
        distinct_layers = [
            row["pipeline_layer"]
            for row in silver_df.select("pipeline_layer").distinct().collect()
        ]
        assert distinct_layers == ["silver"]