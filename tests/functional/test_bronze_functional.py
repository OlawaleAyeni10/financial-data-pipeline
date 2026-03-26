"""
test_bronze_functional.py
-------------------------
Functional tests for the Bronze layer.

Tests the full Bronze layer pipeline end to end —
reading from the actual CSV file and verifying
the output is correct.
"""

import pytest
import os
from src.bronze_layer import (
    run_bronze_layer,
    read_raw_transactions,
    add_ingestion_metadata
)

@pytest.fixture(scope="module")
def bronze_df(spark):
    """Run the full Bronze layer and return the resulting DataFrame."""
    return run_bronze_layer(spark)


class TestBronzeLayerFunctional:

    def test_bronze_reads_all_records(self, bronze_df):
        """Verify Bronze layer reads all records from the CSV."""
        assert bronze_df.count() == 5050

    def test_bronze_output_parquet_exists(self):
        """Verify Bronze layer writes Parquet files to disk."""
        assert os.path.exists("output/bronze/transactions")
        parquet_files = [
            f for f in os.listdir("output/bronze/transactions")
            if f.endswith(".parquet")
        ]
        assert len(parquet_files) > 0

    def test_bronze_has_ingestion_timestamp(self, bronze_df):
        """Verify ingestion_timestamp column is present and non-null."""
        assert "ingestion_timestamp" in bronze_df.columns
        null_count = bronze_df.filter(
            bronze_df.ingestion_timestamp.isNull()
        ).count()
        assert null_count == 0

    def test_bronze_has_source_file(self, bronze_df):
        """Verify source_file column is present and correct."""
        assert "source_file" in bronze_df.columns
        source_files = [
            row["source_file"]
            for row in bronze_df.select("source_file").distinct().collect()
        ]
        assert len(source_files) == 1
        assert "transactions.csv" in source_files[0]

    def test_bronze_pipeline_layer_is_bronze(self, bronze_df):
        """Verify pipeline_layer is set to bronze for all records."""
        distinct_layers = [
            row["pipeline_layer"]
            for row in bronze_df.select("pipeline_layer").distinct().collect()
        ]
        assert distinct_layers == ["bronze"]

    def test_bronze_schema_has_all_expected_columns(self, bronze_df):
        """Verify all expected columns are present in the output."""
        expected_columns = [
            "transaction_id", "customer_id", "merchant_id",
            "merchant_name", "merchant_category", "amount",
            "currency", "channel", "status", "transaction_date",
            "transaction_time", "customer_location", "is_flagged",
            "ingestion_timestamp", "source_file", "pipeline_layer"
        ]
        for col in expected_columns:
            assert col in bronze_df.columns

    def test_bronze_amount_column_is_numeric(self, bronze_df):
        """Verify amount column contains only numeric values."""
        from pyspark.sql.types import DoubleType
        amount_field = next(
            f for f in bronze_df.schema.fields
            if f.name == "amount"
        )
        assert isinstance(amount_field.dataType, DoubleType)

    def test_bronze_preserves_raw_data_including_duplicates(self, bronze_df):
        """
        Verify Bronze layer preserves ALL records including duplicates.
        Cleaning is the Silver layer's responsibility, not Bronze.
        """
        duplicate_count = bronze_df.count() - \
            bronze_df.dropDuplicates(["transaction_id"]).count()
        assert duplicate_count == 50