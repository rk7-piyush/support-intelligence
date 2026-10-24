import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    train_path = DATA_DIR / "train.csv"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    df = pd.read_csv(train_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns.")

    if df[["text", "label"]].isna().any().any():
        raise ValueError("Training data contains missing values.")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Intent classes: {y.nunique()}")

    results = []
    best_model_name = None
    best_macro_f1 = -1.0

    # Compare both algorithms on exactly the same validation split.
    for model_name in ("logistic_regression", "linear_svm"):
        print(f"\nTraining {model_name}...")

        model = build_pipeline(model_name)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": model_name,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = model_name

    # Save validation results for later comparison and reporting.
    results_df = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    )
    results_df.to_csv(
        ARTIFACTS_DIR / "validation_results.csv",
        index=False,
    )

    # Refit the selected algorithm using all available training data.
    print(f"\nSelected model: {best_model_name}")
    print("Retraining selected model on all training examples...")

    best_model = build_pipeline(best_model_name)
    best_model.fit(X, y)

    joblib.dump(
        best_model,
        ARTIFACTS_DIR / "best_model.joblib",
    )

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "validation_macro_f1",
        "validation_results": results,
        "training_rows": int(len(df)),
        "number_of_intents": int(y.nunique()),
        "random_state": 42,
    }

    with open(
        ARTIFACTS_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nValidation comparison:")
    print(results_df.to_string(index=False))
    print("\nSaved model: artifacts/best_model.joblib")
    print("Saved results: artifacts/validation_results.csv")
    print("Saved metadata: artifacts/model_metadata.json")
    print("\nTraining stage completed.")


if __name__ == "__main__":
    main()