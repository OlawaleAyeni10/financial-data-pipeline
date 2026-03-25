"""
generate_data.py
----------------
Generates synthetic financial payment transaction data
simulating a real-world banking/fintech source system.

Intentionally includes:
- Null values (simulating missing data from upstream systems)
- Duplicate transactions (simulating double-processing errors)
- Outlier amounts (simulating potential fraudulent transactions)
- Mixed date formats (simulating data from multiple source systems)
"""

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker('en_GB')
random.seed(42)
Faker.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
NUM_RECORDS = 5000
NUM_CUSTOMERS = 200
NUM_MERCHANTS = 50
OUTPUT_PATH = "data/raw/transactions.csv"

# ── Reference Data ───────────────────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    "Groceries", "Transport", "Dining", "Entertainment",
    "Healthcare", "Utilities", "Retail", "Travel", "ATM Withdrawal"
]

TRANSACTION_STATUSES = ["completed", "pending", "failed", "reversed"]

CURRENCIES = ["GBP", "USD", "EUR"]

CHANNELS = ["online", "in-store", "mobile", "ATM"]


def generate_customer_ids(n):
    """Generate a list of realistic UK-style customer IDs."""
    return [f"CUST-{str(i).zfill(6)}" for i in range(1, n + 1)]


def generate_merchant_ids(n):
    """Generate a list of merchant IDs with associated categories."""
    merchants = []
    for i in range(1, n + 1):
        merchants.append({
            "merchant_id": f"MERCH-{str(i).zfill(4)}",
            "merchant_name": fake.company(),
            "merchant_category": random.choice(MERCHANT_CATEGORIES)
        })
    return merchants


def generate_transaction_amount(category):
    """
    Generate realistic transaction amounts by merchant category.
    Occasionally injects outlier amounts to simulate fraud signals.
    """
    base_ranges = {
        "Groceries": (5, 150),
        "Transport": (2, 80),
        "Dining": (10, 120),
        "Entertainment": (10, 200),
        "Healthcare": (20, 500),
        "Utilities": (50, 300),
        "Retail": (10, 400),
        "Travel": (50, 2000),
        "ATM Withdrawal": (20, 500)
    }
    low, high = base_ranges.get(category, (5, 500))

    # 3% chance of injecting a fraud-like outlier
    if random.random() < 0.03:
        return round(random.uniform(5000, 50000), 2)

    return round(random.uniform(low, high), 2)


def generate_timestamp():
    """Generate a random timestamp within the last 12 months."""
    start = datetime.now() - timedelta(days=365)
    random_seconds = random.randint(0, 365 * 24 * 3600)
    return start + timedelta(seconds=random_seconds)


def generate_transactions(customers, merchants):
    """Generate the full transactions dataset."""
    records = []

    for i in range(1, NUM_RECORDS + 1):
        merchant = random.choice(merchants)
        timestamp = generate_timestamp()
        amount = generate_transaction_amount(merchant["merchant_category"])
        customer_id = random.choice(customers)

        record = {
            "transaction_id": f"TXN-{str(i).zfill(7)}",
            "customer_id": customer_id,
            "merchant_id": merchant["merchant_id"],
            "merchant_name": merchant["merchant_name"],
            "merchant_category": merchant["merchant_category"],
            "amount": amount,
            "currency": random.choice(CURRENCIES),
            "channel": random.choice(CHANNELS),
            "status": random.choice(TRANSACTION_STATUSES),
            "transaction_date": timestamp.strftime("%Y-%m-%d"),
            "transaction_time": timestamp.strftime("%H:%M:%S"),
            "customer_location": fake.city(),
            "is_flagged": amount > 5000  # Flag high-value transactions
        }
        records.append(record)

    return records


def inject_data_quality_issues(df):
    """
    Deliberately inject data quality issues to simulate
    real-world upstream data problems.
    """
    df = df.copy()

    # Inject nulls into ~2% of key fields
    for col in ["customer_id", "merchant_category", "currency", "channel"]:
        null_indices = random.sample(range(len(df)), int(len(df) * 0.02))
        df.loc[null_indices, col] = None

    # Inject ~1% duplicate transactions (double-processing simulation)
    duplicate_indices = random.sample(range(len(df)), int(len(df) * 0.01))
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Inject a small number of negative amounts (data entry errors)
    negative_indices = random.sample(range(len(df)), int(len(df) * 0.005))
    df.loc[negative_indices, "amount"] = df.loc[negative_indices, "amount"] * -1

    return df


def main():
    print("Generating synthetic payment transactions dataset...")

    customers = generate_customer_ids(NUM_CUSTOMERS)
    merchants = generate_merchant_ids(NUM_MERCHANTS)

    records = generate_transactions(customers, merchants)
    df = pd.DataFrame(records)
    df = inject_data_quality_issues(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset generated successfully:")
    print(f"  Total records : {len(df)}")
    print(f"  Null values   : {df.isnull().sum().sum()}")
    print(f"  Duplicates    : {df.duplicated().sum()}")
    print(f"  Flagged txns  : {df['is_flagged'].sum()}")
    print(f"  Saved to      : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()