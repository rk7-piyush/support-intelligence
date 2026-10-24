from pathlib import Path
from io import BytesIO

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_URLS = {
    "train": (
        "https://raw.githubusercontent.com/"
        "PolyAI-LDN/task-specific-datasets/master/"
        "banking_data/train.csv"
    ),
    "test": (
        "https://raw.githubusercontent.com/"
        "PolyAI-LDN/task-specific-datasets/master/"
        "banking_data/test.csv"
    ),
}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, url in DATA_URLS.items():
        print(f"\nDownloading {split_name} dataset...")

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        df = pd.read_csv(BytesIO(response.content))

        # Normalize the original source's category column name.
        if "category" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"category": "label"})

        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(
                f"Unexpected columns in {split_name}: "
                f"{list(df.columns)}"
            )

        if df[["text", "label"]].isna().any().any():
            raise ValueError(f"Missing text or labels in {split_name}.")

        if len(df) != {"train": 10003, "test": 3080}[split_name]:
            raise ValueError(
                f"Unexpected row count in {split_name}: {len(df)}"
            )

        output_path = DATA_DIR / f"{split_name}.csv"
        df[["text", "label"]].to_csv(output_path, index=False)

        print(f"Saved to: {output_path}")
        print(f"Rows: {len(df)}")
        print(f"Unique intents: {df['label'].nunique()}")
        print(f"Columns: {list(df.columns)}")

    print("\nDataset download and validation completed.")


if __name__ == "__main__":
    main()