"""
test_bronze.py
--------------
Unit tests for the Bronze layer.

Tests each function in bronze_layer.py in isolation
using a small synthetic DataFrame rather than reading
from disk.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType
)
from src.bronze_layer import add_ingestion_metadata, RAW_SCHEMA


@pytest.fixture(scope="module")
def spark():
    """
    Create a single shared SparkSession for all unit tests.
    Using scope="module" means Spark starts once per test
    file rather than once per test — significantly faster.
    """
    spark = SparkSession.builder \
        .appName("BronzeUnitTests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_bronze_df(spark):
    """
    Create a small sample DataFrame that mirrors
    the raw transaction schema for testing purposes.
    """
    data = [
        ("TXN-0000001", "CUST-000001", "MERCH-0001", "Test Merchant",
         "Groceries", 45.50, "GBP", "online", "completed",
         "2025-06-01", "10:30:00", "London", False),
        ("TXN-0000002", "CUST-000002", "MERCH-0002", "Another Merchant",
         "Transport", 12.00, "USD", "mobile", "pending",
         "2025-06-02", "14:15:00", "Manchester", False),
        ("TXN-0000003", "CUST-000003", "MERCH-0003", "Third Merchant",
         "Dining", 8500.00, "EUR", "in-store", "completed",
         "2025-06-03", "19:45:00", "Birmingham", True),
    ]
    return spark.createDataFrame(data, schema=RAW_SCHEMA)


class TestAddIngestionMetadata:

    def test_adds_ingestion_timestamp_column(self, sample_bronze_df):
        """Verify ingestion_timestamp column is added."""
        df = add_ingestion_metadata(sample_bronze_df)
        assert "ingestion_timestamp" in df.columns

    def test_adds_source_file_column(self, sample_bronze_df):
        """Verify source_file column is added."""
        df = add_ingestion_metadata(sample_bronze_df)
        assert "source_file" in df.columns

    def test_adds_pipeline_layer_column(self, sample_bronze_df):
        """Verify pipeline_layer column is added."""
        df = add_ingestion_metadata(sample_bronze_df)
        assert "pipeline_layer" in df.columns

    def test_pipeline_layer_value_is_bronze(self, sample_bronze_df):
        """Verify pipeline_layer is set to 'bronze'."""
        df = add_ingestion_metadata(sample_bronze_df)
        layers = [row["pipeline_layer"] for row in df.collect()]
        assert all(layer == "bronze" for layer in layers)

    def test_row_count_unchanged(self, sample_bronze_df):
        """Verify metadata addition does not drop or add rows."""
        df = add_ingestion_metadata(sample_bronze_df)
        assert df.count() == sample_bronze_df.count()

    def test_original_columns_preserved(self, sample_bronze_df):
        """Verify all original columns are still present."""
        df = add_ingestion_metadata(sample_bronze_df)
        for col in sample_bronze_df.columns:
            assert col in df.columns


class TestRawSchema:

    def test_schema_has_correct_field_count(self):
        """Verify schema has expected number of fields."""
        assert len(RAW_SCHEMA.fields) == 13

    def test_schema_contains_transaction_id(self):
        """Verify transaction_id is in the schema."""
        field_names = [f.name for f in RAW_SCHEMA.fields]
        assert "transaction_id" in field_names

    def test_amount_field_is_double_type(self):
        """Verify amount field is DoubleType."""
        amount_field = next(
            f for f in RAW_SCHEMA.fields if f.name == "amount"
        )
        assert isinstance(amount_field.dataType, DoubleType)

    def test_is_flagged_field_is_boolean_type(self):
        """Verify is_flagged field is BooleanType."""
        flagged_field = next(
            f for f in RAW_SCHEMA.fields if f.name == "is_flagged"
        )
        assert isinstance(flagged_field.dataType, BooleanType)