"""
test_pipeline_integration.py
-----------------------------
Integration tests for the full pipeline.

Tests the complete Bronze → Silver → Gold pipeline
end to end, verifying that all layers work together
correctly and produce consistent, expected outputs.

Integration tests are broader than functional tests —
they test the pipeline as a whole system rather than
individual layers in isolation.
"""

import pytest
import os
from pyspark.sql import functions as F
from src.bronze_layer import run_bronze_layer
from src.silver_layer import run_silver_layer
from src.gold_layer import run_gold_layer


@pytest.fixture(scope="module")
def pipeline_outputs(spark):
    """
    Run the complete pipeline end to end and return
    all layer outputs for integration assertions.
    """
    df_bronze = run_bronze_layer(spark)
    df_silver = run_silver_layer(spark, df_bronze)
    gold_tables = run_gold_layer(spark, df_silver)

    return {
        "bronze": df_bronze,
        "silver": df_silver,
        "gold": gold_tables
    }


class TestPipelineDataFlow:

    def test_bronze_to_silver_record_reduction(self, pipeline_outputs):
        """
        Verify Silver has fewer records than Bronze.
        Bronze preserves everything; Silver cleans.
        """
        bronze_count = pipeline_outputs["bronze"].count()
        silver_count = pipeline_outputs["silver"].count()
        assert silver_count < bronze_count

    def test_silver_removes_exactly_175_records(self, pipeline_outputs):
        """
        Verify Silver removes exactly the expected number
        of records: 50 duplicates + 100 critical nulls
        + 25 invalid amounts = 175 total.
        """
        bronze_count = pipeline_outputs["bronze"].count()
        silver_count = pipeline_outputs["silver"].count()
        assert bronze_count - silver_count == 175

    def test_all_silver_records_appear_in_bronze(self, pipeline_outputs):
        """
        Verify every transaction_id in Silver exists in Bronze.
        Silver should only remove records, never add new ones.
        """
        bronze_ids = set(
            row["transaction_id"]
            for row in pipeline_outputs["bronze"]
            .select("transaction_id").collect()
        )
        silver_ids = set(
            row["transaction_id"]
            for row in pipeline_outputs["silver"]
            .select("transaction_id").collect()
        )
        assert silver_ids.issubset(bronze_ids)

    def test_gold_customer_ids_exist_in_silver(self, pipeline_outputs):
        """
        Verify all customer_ids in the Gold customer summary
        exist in the Silver layer — no phantom customers.
        """
        silver_customers = set(
            row["customer_id"]
            for row in pipeline_outputs["silver"]
            .select("customer_id").distinct().collect()
        )
        gold_customers = set(
            row["customer_id"]
            for row in pipeline_outputs["gold"]["customer_summary"]
            .select("customer_id").collect()
        )
        assert gold_customers.issubset(silver_customers)

    def test_gold_merchant_ids_exist_in_silver(self, pipeline_outputs):
        """
        Verify all merchant_ids in Gold merchant analytics
        exist in the Silver layer — no phantom merchants.
        """
        silver_merchants = set(
            row["merchant_id"]
            for row in pipeline_outputs["silver"]
            .select("merchant_id").distinct().collect()
        )
        gold_merchants = set(
            row["merchant_id"]
            for row in pipeline_outputs["gold"]["merchant_analytics"]
            .select("merchant_id").collect()
        )
        assert gold_merchants.issubset(silver_merchants)

    def test_gold_months_exist_in_silver(self, pipeline_outputs):
        """
        Verify all months in Gold monthly trends exist
        in the Silver layer — no phantom months.
        """
        silver_months = set(
            row["transaction_month"]
            for row in pipeline_outputs["silver"]
            .select("transaction_month").distinct().collect()
        )
        gold_months = set(
            row["transaction_month"]
            for row in pipeline_outputs["gold"]["monthly_trends"]
            .select("transaction_month").collect()
        )
        assert gold_months.issubset(silver_months)


class TestPipelineDataConsistency:

    def test_total_silver_volume_matches_gold_monthly_total(
        self, pipeline_outputs
    ):
        """
        Verify total transaction volume in Silver matches
        the sum of monthly volumes in Gold monthly trends.
        This is a critical consistency check — if these
        numbers don't match, something has been lost or
        duplicated between layers.
        """
        silver_total = pipeline_outputs["silver"] \
            .agg(F.round(F.sum("amount"), 2)) \
            .collect()[0][0]

        gold_total = pipeline_outputs["gold"]["monthly_trends"] \
            .agg(F.round(F.sum("total_volume"), 2)) \
            .collect()[0][0]

        assert silver_total == gold_total

    def test_silver_transaction_count_matches_gold_monthly_count(
        self, pipeline_outputs
    ):
        """
        Verify total transaction count in Silver matches
        the sum of monthly transaction counts in Gold.
        """
        silver_count = pipeline_outputs["silver"].count()

        gold_count = pipeline_outputs["gold"]["monthly_trends"] \
            .agg(F.sum("total_transactions")) \
            .collect()[0][0]

        assert silver_count == gold_count

    def test_gold_customer_total_spend_matches_silver(
        self, pipeline_outputs
    ):
        """
        Verify sum of all customer total_spend in Gold
        matches total Silver transaction volume.
        """
        gold_customer_total = pipeline_outputs["gold"]["customer_summary"] \
            .agg(F.round(F.sum("total_spend"), 2)) \
            .collect()[0][0]

        silver_total = pipeline_outputs["silver"] \
            .agg(F.round(F.sum("amount"), 2)) \
            .collect()[0][0]

        assert gold_customer_total == silver_total

    def test_flagged_transactions_consistent_across_layers(
        self, pipeline_outputs
    ):
        """
        Verify flagged transaction count is consistent
        between Silver and Gold customer summary.
        """
        silver_flagged = pipeline_outputs["silver"] \
            .filter(F.col("is_flagged") == True) \
            .count()

        gold_flagged = pipeline_outputs["gold"]["customer_summary"] \
            .agg(F.sum("flagged_transaction_count")) \
            .collect()[0][0]

        assert silver_flagged == gold_flagged


class TestPipelineOutputFiles:

    def test_all_output_directories_exist(self):
        """Verify all expected output directories were created."""
        expected_paths = [
            "output/bronze/transactions",
            "output/silver/transactions",
            "output/gold/customer_summary",
            "output/gold/merchant_analytics",
            "output/gold/monthly_trends"
        ]
        for path in expected_paths:
            assert os.path.exists(path), f"Missing output path: {path}"

    def test_all_output_directories_contain_parquet_files(self):
        """Verify all output directories contain Parquet files."""
        paths_to_check = [
            "output/bronze/transactions",
            "output/gold/customer_summary",
            "output/gold/merchant_analytics",
            "output/gold/monthly_trends"
        ]
        for path in paths_to_check:
            all_files = []
            for root, dirs, files in os.walk(path):
                all_files.extend(
                    f for f in files if f.endswith(".parquet")
                )
            assert len(all_files) > 0, \
                f"No Parquet files found in: {path}"

    def test_silver_output_is_partitioned_by_month(self):
        """Verify Silver output is partitioned by transaction_month."""
        partitions = [
            f for f in os.listdir("output/silver/transactions")
            if f.startswith("transaction_month=")
        ]
        assert len(partitions) > 0


class TestPipelineBusinessRules:

    def test_no_completed_transactions_lost(self, pipeline_outputs):
        """
        Verify the pipeline does not drop any completed
        transactions — only duplicates, nulls, and invalid
        amounts should be removed.
        """
        silver_completed = pipeline_outputs["silver"] \
            .filter(F.col("status") == "completed") \
            .count()

        gold_completed = pipeline_outputs["gold"]["customer_summary"] \
            .agg(F.sum("completed_transactions")) \
            .collect()[0][0]

        assert silver_completed == gold_completed

    def test_premium_customers_have_high_spend(self, pipeline_outputs):
        """
        Verify all Premium customers have total_spend >= 50000
        as defined by the business segmentation rules.
        """
        premium_customers = pipeline_outputs["gold"]["customer_summary"] \
            .filter(F.col("customer_segment") == "Premium")

        low_spend_premium = premium_customers \
            .filter(F.col("total_spend") < 50000) \
            .count()

        assert low_spend_premium == 0

    def test_high_risk_merchants_have_high_flagged_ratio(
        self, pipeline_outputs
    ):
        """
        Verify all High Risk merchants have a flagged
        transaction ratio above 5% as per business rules.
        """
        high_risk = pipeline_outputs["gold"]["merchant_analytics"] \
            .filter(F.col("merchant_risk_flag") == "High Risk") \
            .withColumn(
                "flagged_ratio",
                F.col("flagged_transactions") /
                F.col("total_transactions")
            )

        low_ratio_high_risk = high_risk \
            .filter(F.col("flagged_ratio") <= 0.05) \
            .count()

        assert low_ratio_high_risk == 0

    def test_running_total_final_value_matches_total_volume(
        self, pipeline_outputs
    ):
        """
        Verify the final running total in monthly trends
        equals the total transaction volume — confirming
        the cumulative calculation is correct end to end.
        """
        final_running_total = pipeline_outputs["gold"]["monthly_trends"] \
            .orderBy(F.col("transaction_month").desc()) \
            .select("running_total_volume") \
            .collect()[0][0]

        total_volume = pipeline_outputs["gold"]["monthly_trends"] \
            .agg(F.sum("total_volume")) \
            .collect()[0][0]

        assert round(final_running_total, 2) == round(total_volume, 2)