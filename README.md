# Financial Transactions Data Pipeline

A production-style PySpark data pipeline implementing the **Medallion Architecture** 
(Bronze → Silver → Gold) for financial payment transaction analytics.

Built to simulate the kind of data engineering work done in financial services 
environments — processing payment transactions, detecting anomalies, and producing 
business-ready analytics datasets.

---

## Architecture
```
Source Data (CSV / API / Core Banking System)
        ↓
Bronze Layer  — Raw ingestion, schema enforcement, metadata tagging
        ↓
Silver Layer  — Deduplication, validation, cleaning, standardisation
        ↓
Gold Layer    — Customer summaries, merchant analytics, monthly trends
        ↓
Consumers     — Power BI / Azure ML / Downstream APIs
```

In a production Azure environment this pipeline would run on **Azure Databricks**, 
reading from and writing to **Azure Data Lake Storage Gen2**, with tables registered 
in **Unity Catalog** and orchestrated via **Azure Data Factory**.

---

## Project Structure
```
financial-data-pipeline/
│
├── data/
│   └── raw/                        # Landing zone (raw CSV)
│
├── output/
│   ├── bronze/                     # Raw Parquet
│   ├── silver/                     # Cleaned Parquet (partitioned by month)
│   └── gold/                       # Business aggregations
│       ├── customer_summary/
│       ├── merchant_analytics/
│       └── monthly_trends/
│
├── src/
│   ├── bronze_layer.py             # Raw ingestion
│   ├── silver_layer.py             # Cleaning & validation
│   └── gold_layer.py               # Business aggregations
│
├── tests/
│   ├── unit/                       # Unit tests per layer
│   ├── functional/                 # Functional tests per layer
│   └── integration/                # Full pipeline integration tests
│
├── generate_data.py                # Synthetic data generator
├── main.py                         # Pipeline orchestrator
└── requirements.txt                # Python dependencies
```

---

## Pipeline Layers

### Bronze — Raw Ingestion
- Reads raw CSV from the landing zone
- Enforces an explicit schema (no schema inference)
- Adds ingestion metadata: timestamp, source file, pipeline layer
- Writes to Parquet — no transformations applied
- Preserves raw data for reprocessing if downstream logic changes

### Silver — Cleaning & Validation
- Removes duplicate transactions (50 duplicates removed from sample dataset)
- Drops records with nulls in critical fields (transaction_id, customer_id, amount)
- Removes invalid amounts (negative or zero values)
- Standardises categorical fields (currency, channel, status)
- Combines transaction_date and transaction_time into a single timestamp
- Partitions output by transaction_month for query performance
- Retains non-critical nulls (e.g. customer_location) for downstream handling

### Gold — Business Aggregations
Produces three business-ready tables:

**customer_summary** — one row per customer containing:
- Total and average spend, transaction count
- Customer segment (Premium / High Value / Mid Tier / Standard)
- Fraud risk score (% of flagged transactions)
- Preferred channel and merchant category
- First and last transaction dates

**merchant_analytics** — one row per merchant containing:
- Total revenue and transaction volume
- Unique customer count
- Merchant risk flag (High / Medium / Low)
- Rank within merchant category by revenue

**monthly_trends** — one row per month containing:
- Total transaction volume and count
- Active customer and merchant counts
- Month-over-month volume growth %
- Running total volume (cumulative)

---

## Sample Pipeline Output
```
Total customers processed : 237
Total merchants analysed  : 50
Months of data covered    : 13
High risk merchants       : 6
Premium customers         : 37
Total transaction volume  : £4,699,772.95
Pipeline runtime          : ~9 seconds
```

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14 | Core language |
| PySpark | 4.1.1 | Distributed data processing |
| Java JDK | 17 | PySpark runtime requirement |
| Pandas | 2.2.3 | Synthetic data generation |
| Faker | 37.1.0 | Realistic test data generation |
| pytest | 8.3.5 | Unit, functional and integration tests |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Java JDK 17
- Git

### Setup
```bash
# Clone the repository
git clone git@github.com:OlawaleAyeni10/financial-data-pipeline.git
cd financial-data-pipeline

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Generate synthetic data
python generate_data.py

# Run the full pipeline
python main.py
```

### Run the Tests
```bash
pytest tests/ -v
```

---

## Key Design Decisions

**Explicit schema enforcement over schema inference** — defined upfront in the Bronze 
layer to catch upstream changes early and avoid silent type coercions.

**Medallion Architecture** — separating raw, cleaned, and aggregated data into distinct 
layers provides clear data lineage, simplifies reprocessing, and makes the pipeline 
easier to maintain and debug.

**Partitioning by transaction_month** — the Silver layer partitions output by month, 
reflecting real-world query patterns in financial services where most analysis is 
time-period based. This eliminates unnecessary file scans and improves query performance.

**Retaining non-critical nulls** — rather than dropping all null records at the Silver 
layer, only records missing critical fields are removed. Non-critical nulls are retained 
and handled at the Gold layer, preserving as much data as possible.

**Fraud risk scoring at the customer level** — the Gold layer calculates a fraud risk 
score per customer based on the ratio of flagged transactions. In a production environment 
this would feed into an Azure ML model for more sophisticated anomaly detection.

---

## Azure Production Architecture

In a real client engagement this pipeline would be deployed as follows:

- **Ingestion** — Azure Data Factory pulls raw data from source systems into ADLS Gen2
- **Processing** — Databricks Jobs run Bronze → Silver → Gold on a scheduled cluster
- **Storage** — Azure Data Lake Storage Gen2 with hierarchical namespace enabled
- **Governance** — Unity Catalog for table registration, lineage, and access control
- **Monitoring** — Azure Monitor + custom pipeline metrics for observability
- **Consumption** — Power BI connects to Gold layer tables via Databricks SQL Warehouse