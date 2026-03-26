"""
test_silver.py
--------------
Unit tests for the Silver layer.

Tests each cleaning and validation function in isolation
to verify correct behaviour on known input data.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType, TimestampType
)
from src.silver_layer import (
    remove_duplicates,
    drop_critical_nulls,
    validate_amounts,
    standardise_categoricals,
    build_transaction_timestamp,
    CRITICAL_FIELDS,
    VALID_CURRENCIES,
    VALID_CHANNELS,
    VALID_STATUSES
)


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("SilverUnitTests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_schema():
    return StructType([
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
        StructField("ingestion_timestamp", StringType(), True),
        StructField("source_file",       StringType(),  True),
        StructField("pipeline_layer",    StringType(),  True),
    ])


@pytest.fixture(scope="module")
def sample_df(spark, sample_schema):
    data = [
        ("TXN-001", "CUST-001", "MERCH-001", "Merchant A",
         "Groceries", 50.0, "GBP", "online", "completed",
         "2025-06-01", "10:00:00", "London", False, None, None, "bronze"),
        ("TXN-002", "CUST-002", "MERCH-002", "Merchant B",
         "Transport", 20.0, "USD", "mobile", "pending",
         "2025-06-02", "11:00:00", "Manchester", False, None, None, "bronze"),
        ("TXN-002", "CUST-002", "MERCH-002", "Merchant B",
         "Transport", 20.0, "USD", "mobile", "pending",
         "2025-06-02", "11:00:00", "Manchester", False, None, None, "bronze"),
        ("TXN-003", None, "MERCH-003", "Merchant C",
         "Dining", 30.0, "EUR", "in-store", "completed",
         "2025-06-03", "12:00:00", "Birmingham", False, None, None, "bronze"),
        ("TXN-004", "CUST-004", "MERCH-004", "Merchant D",
         "Retail", -15.0, "GBP", "online", "completed",
         "2025-06-04", "13:00:00", "Leeds", False, None, None, "bronze"),
        ("TXN-005", "CUST-005", "MERCH-005", "Merchant E",
         "Entertainment", 75.0, "INVALID", "INVALID", "INVALID",
         "2025-06-05", "14:00:00", "Bristol", False, None, None, "bronze"),
    ]
    return spark.createDataFrame(data, schema=sample_schema)


class TestRemoveDuplicates:

    def test_removes_duplicate_transaction_ids(self, sample_df):
        """Verify duplicate transaction_ids are removed."""
        df = remove_duplicates(sample_df)
        ids = [row["transaction_id"] for row in df.collect()]
        assert len(ids) == len(set(ids))

    def test_correct_record_count_after_dedup(self, sample_df):
        """Verify correct number of records after deduplication."""
        df = remove_duplicates(sample_df)
        assert df.count() == 5

    def test_original_columns_preserved(self, sample_df):
        """Verify no columns are dropped during deduplication."""
        df = remove_duplicates(sample_df)
        assert df.columns == sample_df.columns


class TestDropCriticalNulls:

    def test_drops_records_with_null_customer_id(self, sample_df):
        """Verify records with null customer_id are dropped."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        customer_ids = [row["customer_id"] for row in df.collect()]
        assert None not in customer_ids

    def test_correct_record_count_after_null_drop(self, sample_df):
        """Verify correct record count after dropping critical nulls."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        assert df.count() == 4

    def test_critical_fields_defined(self):
        """Verify critical fields list is not empty."""
        assert len(CRITICAL_FIELDS) > 0
        assert "transaction_id" in CRITICAL_FIELDS
        assert "customer_id" in CRITICAL_FIELDS
        assert "amount" in CRITICAL_FIELDS


class TestValidateAmounts:

    def test_removes_negative_amounts(self, sample_df):
        """Verify records with negative amounts are removed."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        amounts = [row["amount"] for row in df.collect()]
        assert all(a > 0 for a in amounts)

    def test_correct_record_count_after_amount_validation(self, sample_df):
        """Verify correct record count after amount validation."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        assert df.count() == 3


class TestStandardiseCategoricals:

    def test_invalid_currency_replaced_with_unknown(self, sample_df):
        """Verify invalid currency values are replaced with 'unknown'."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        currencies = [row["currency"] for row in df.collect()]
        for currency in currencies:
            assert currency in VALID_CURRENCIES + ["unknown"]

    def test_invalid_channel_replaced_with_unknown(self, sample_df):
        """Verify invalid channel values are replaced with 'unknown'."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        channels = [row["channel"] for row in df.collect()]
        for channel in channels:
            assert channel in VALID_CHANNELS + ["unknown"]

    def test_invalid_status_replaced_with_unknown(self, sample_df):
        """Verify invalid status values are replaced with 'unknown'."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        statuses = [row["status"] for row in df.collect()]
        for status in statuses:
            assert status in VALID_STATUSES + ["unknown"]


class TestBuildTransactionTimestamp:

    def test_adds_transaction_timestamp_column(self, sample_df):
        """Verify transaction_timestamp column is added."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        df = build_transaction_timestamp(df)
        assert "transaction_timestamp" in df.columns

    def test_adds_transaction_month_column(self, sample_df):
        """Verify transaction_month column is added."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        df = build_transaction_timestamp(df)
        assert "transaction_month" in df.columns

    def test_adds_day_of_week_column(self, sample_df):
        """Verify day_of_week column is added."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        df = build_transaction_timestamp(df)
        assert "day_of_week" in df.columns

    def test_transaction_month_format(self, sample_df):
        """Verify transaction_month follows yyyy-MM format."""
        df = remove_duplicates(sample_df)
        df = drop_critical_nulls(df)
        df = validate_amounts(df)
        df = standardise_categoricals(df)
        df = build_transaction_timestamp(df)
        months = [row["transaction_month"] for row in df.collect()]
        for month in months:
            assert len(month) == 7
            assert month[4] == "-"