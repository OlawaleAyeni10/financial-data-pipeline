"""
conftest.py
-----------
Shared pytest fixtures for all test modules.

Using a single shared SparkSession across all tests via
conftest.py is the correct approach for PySpark testing —
it avoids the overhead of starting multiple JVM instances
and prevents port binding conflicts on local machines.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Create a single shared SparkSession for the entire
    test session. scope="session" means this SparkSession
    is created once and reused across all test files.
    """
    spark = SparkSession.builder \
        .appName("FinancialPipelineTests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()