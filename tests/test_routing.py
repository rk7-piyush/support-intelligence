import pytest

from src.routing import route_ticket, get_department


class FakeModel:
    """Small deterministic model for testing routing behavior."""

    classes_ = ["card_arrival", "cash_withdrawal", "transfer_timing"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, texts):
        return [self.probabilities]


@pytest.fixture
def settings():
    return {"confidence_threshold": 0.55}


@pytest.fixture
def config():
    return {
        "default_department": "General Support",
        "manual_review_department": "Manual Review Queue",
        "rules": [
            {
                "department": "Account and Balance",
                "keywords": ["balance"],
            },
            {
                "department": "Transfers and Payments",
                "keywords": ["transfer"],
            },
            {
                "department": "Card Services",
                "keywords": ["card"],
            },
        ],
    }


def test_high_confidence_ticket_is_auto_routed(settings, config):
    model = FakeModel([0.80, 0.10, 0.10])

    result = route_ticket(
        "My card has not arrived",
        model=model,
        settings=settings,
        config=config,
    )

    assert result["decision"] == "AUTO_ROUTED"
    assert result["assigned_department"] == "Card Services"
    assert result["predicted_intent"] == "card_arrival"
    assert result["confidence"] == pytest.approx(0.80)


def test_low_confidence_ticket_goes_to_review(settings, config):
    model = FakeModel([0.40, 0.35, 0.25])

    result = route_ticket(
        "I need help",
        model=model,
        settings=settings,
        config=config,
    )

    assert result["decision"] == "MANUAL_REVIEW"
    assert result["assigned_department"] == "Manual Review Queue"
    assert result["confidence"] == pytest.approx(0.40)


def test_exact_threshold_is_auto_routed(settings, config):
    model = FakeModel([0.55, 0.25, 0.20])

    result = route_ticket(
        "My card has not arrived",
        model=model,
        settings=settings,
        config=config,
    )

    assert result["decision"] == "AUTO_ROUTED"


def test_empty_ticket_is_rejected(settings, config):
    model = FakeModel([0.40, 0.35, 0.25])

    with pytest.raises(ValueError):
        route_ticket(
            "   ",
            model=model,
            settings=settings,
            config=config,
        )


def test_specific_balance_rule_wins(config):
    department = get_department(
        "balance_not_updated_after_bank_transfer",
        config,
    )

    assert department == "Account and Balance"


def test_unknown_intent_uses_default_department():
    config = {
        "default_department": "General Support",
        "rules": [],
    }

    assert get_department("unknown_intent", config) == "General Support"