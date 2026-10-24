import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_AUTO_ROUTE_ACCURACY = 0.95
MIN_AUTO_ROUTE_COVERAGE = 0.05


def build_calibrated_model():
    return CalibratedClassifierCV(
        estimator=build_pipeline("linear_svm"),
        method="sigmoid",
        cv=3,
    )


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Fitting calibrated model for threshold selection...")
    validation_model = build_calibrated_model()
    validation_model.fit(X_fit, y_fit)

    probabilities = validation_model.predict_proba(X_val)
    class_names = validation_model.classes_

    predicted_indices = probabilities.argmax(axis=1)
    predictions = class_names[predicted_indices]
    max_confidence = probabilities.max(axis=1)

    print(
        "Validation accuracy without a review threshold:",
        f"{accuracy_score(y_val, predictions):.4f}",
    )

    rows = []

    for threshold in np.arange(0.30, 0.96, 0.05):
        accepted = max_confidence >= threshold
        coverage = float(accepted.mean())

        if accepted.any():
            accepted_accuracy = accuracy_score(
                y_val.iloc[np.flatnonzero(accepted)],
                predictions[accepted],
            )
        else:
            accepted_accuracy = float("nan")

        rows.append({
            "threshold": round(float(threshold), 2),
            "coverage": coverage,
            "review_rate": 1.0 - coverage,
            "auto_route_accuracy": float(accepted_accuracy),
            "auto_routed_count": int(accepted.sum()),
        })

    results = pd.DataFrame(rows)

    print("\nValidation threshold analysis:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    eligible = results[
        (results["auto_route_accuracy"] >= TARGET_AUTO_ROUTE_ACCURACY)
        & (results["coverage"] >= MIN_AUTO_ROUTE_COVERAGE)
    ]

    if not eligible.empty:
        # Highest coverage that meets the target accuracy.
        chosen = eligible.sort_values(
            "coverage", ascending=False
        ).iloc[0]
        target_met = True
    else:
        # No threshold met both targets. Prefer the most accurate
        # observed option; do not claim the target was achieved.
        candidates = results[
            results["auto_routed_count"] > 0
        ].sort_values(
            ["auto_route_accuracy", "coverage"],
            ascending=[False, False],
        )
        chosen = candidates.iloc[0]
        target_met = (
            chosen["auto_route_accuracy"]
            >= TARGET_AUTO_ROUTE_ACCURACY
        )

    threshold = float(chosen["threshold"])

    print(f"\nSelected threshold: {threshold:.2f}")
    print(f"Validation coverage: {chosen['coverage']:.4f}")
    print(
        "Validation accuracy among auto-routed examples:",
        f"{chosen['auto_route_accuracy']:.4f}",
    )
    print(f"95% target met: {target_met}")

    # Refit the calibrated model on the complete training dataset.
    print("\nFitting final calibrated model on all training data...")
    final_model = build_calibrated_model()
    final_model.fit(X, y)

    joblib.dump(
        final_model,
        ARTIFACTS_DIR / "best_model.joblib",
    )

    settings = {
        "selected_model": "calibrated_linear_svm",
        "confidence_threshold": threshold,
        "threshold_selection_metric": "validation_accuracy_at_coverage",
        "target_auto_route_accuracy": TARGET_AUTO_ROUTE_ACCURACY,
        "target_met_on_validation": bool(target_met),
        "validation_coverage": float(chosen["coverage"]),
        "validation_auto_route_accuracy": float(
            chosen["auto_route_accuracy"]
        ),
        "validation_examples": int(len(X_val)),
        "calibration_method": "sigmoid",
        "calibration_cv_folds": 3,
    }

    with open(
        ARTIFACTS_DIR / "routing_settings.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(settings, file, indent=2)

    results.to_csv(
        ARTIFACTS_DIR / "threshold_analysis.csv",
        index=False,
    )

    print("\nSaved calibrated model: artifacts/best_model.joblib")
    print("Saved routing settings: artifacts/routing_settings.json")
    print("Saved threshold analysis: artifacts/threshold_analysis.csv")


if __name__ == "__main__":
    main()