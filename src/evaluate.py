import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    test_path = DATA_DIR / "test.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    if test_df[["text", "label"]].isna().any().any():
        raise ValueError("Test data contains missing values.")

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    print("Evaluating saved model on untouched test data...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # Save metrics for every intent.
    per_class = pd.DataFrame(report).T
    per_class.index.name = "intent"
    per_class.to_csv(ARTIFACTS_DIR / "per_class_metrics.csv")

    # Save the confusion matrix with explicit intent labels.
    labels = sorted(set(y_test) | set(predictions))
    matrix = confusion_matrix(
        y_test, predictions, labels=labels
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    matrix_df.index.name = "actual_intent"
    matrix_df.to_csv(ARTIFACTS_DIR / "confusion_matrix.csv")

    # Save individual predictions for later error analysis.
    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = predictions
    predictions_df["correct"] = (
        predictions_df["label"] == predictions_df["predicted_label"]
    )
    predictions_df.to_csv(
        ARTIFACTS_DIR / "test_predictions.csv",
        index=False,
    )

    summary = {
        "model": "Linear SVM or selected pipeline",
        "test_examples": int(len(test_df)),
        "number_of_intents": int(y_test.nunique()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    with open(
        ARTIFACTS_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # Display the weakest intents to guide future error analysis.
    class_metrics = per_class.loc[
        per_class.index.isin(labels)
    ].sort_values("f1-score")

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test examples: {len(test_df)}")
    print(f"Intent classes: {y_test.nunique()}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}")
    print(f"Weighted-F1:    {weighted_f1:.4f}")

    print("\n=== 10 LOWEST-F1 INTENTS ===")
    print(
        class_metrics[
            ["precision", "recall", "f1-score", "support"]
        ].head(10).to_string()
    )

    print("\nSaved evaluation artifacts:")
    print("- artifacts/evaluation_summary.json")
    print("- artifacts/per_class_metrics.csv")
    print("- artifacts/confusion_matrix.csv")
    print("- artifacts/test_predictions.csv")


if __name__ == "__main__":
    main()