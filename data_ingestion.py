from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR
INGESTED_DIR = BASE_DIR / "ingested"

# Define files
INPUT_FEATURES = RAW_DIR / "A.csv"
INPUT_TARGETS = RAW_DIR / "A_targets.csv"
OUTPUT_FILE = INGESTED_DIR / "A_join.csv"


def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data (fitur & target terpisah untuk Dataset A)
    df_features = pd.read_csv(INPUT_FEATURES)
    df_targets = pd.read_csv(INPUT_TARGETS)

    # Merge features & targets berdasarkan Student_ID
    df = pd.merge(df_features, df_targets, on="Student_ID")

    # Basic validation
    assert not df.empty, "Dataset is empty"
    assert "placement_status" in df.columns, "Target placement_status tidak ditemukan"
    assert "salary_lpa" in df.columns, "Target salary_lpa tidak ditemukan"

    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Data ingested from {INPUT_FEATURES} + {INPUT_TARGETS} → {OUTPUT_FILE}")
    print(f"   Shape: {df.shape}")


if __name__ == "__main__":
    ingest_data()