import json
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = PROJECT_ROOT / "config" / "intent_routing.json"


def load_routing_settings():
    with open(
        ARTIFACTS_DIR / "routing_settings.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_routing_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_department(intent, config):
    """Map an intent label to a department using ordered rules."""
    normalized_intent = intent.lower()

    for rule in config["rules"]:
        for keyword in rule["keywords"]:
            if keyword.lower() in normalized_intent:
                return rule["department"]

    return config["default_department"]


def route_ticket(text, model=None, settings=None, config=None):
    """
    Predict a ticket's intent and decide whether it needs manual review.

    Confidence is the calibrated probability assigned to the
    highest-probability intent. It is not a guarantee of correctness.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text must be a non-empty string.")

    if model is None:
        model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")

    if settings is None:
        settings = load_routing_settings()

    if config is None:
        config = load_routing_config()

    probabilities = model.predict_proba([text])[0]
    best_index = max(
    range(len(probabilities)),
    key=lambda index: probabilities[index],
)

    predicted_intent = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    threshold = float(settings["confidence_threshold"])

    proposed_department = get_department(predicted_intent, config)

    if confidence >= threshold:
        decision = "AUTO_ROUTED"
        assigned_department = proposed_department
        reason = "Confidence meets or exceeds the configured threshold."
    else:
        decision = "MANUAL_REVIEW"
        assigned_department = config["manual_review_department"]
        reason = "Confidence is below the configured threshold."

    return {
        "ticket_text": text.strip(),
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "proposed_department": proposed_department,
        "assigned_department": assigned_department,
        "decision": decision,
        "reason": reason,
    }