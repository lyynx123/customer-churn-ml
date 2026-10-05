from pathlib import Path

# Base directory for project
BASE_DIR = Path(__file__).resolve().parents[2]

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_processed.parquet"

# Random seed for reproducibility
RANDOM_SEED = 42
