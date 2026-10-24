"""Triage for long customer letters.

Pipeline: clean boilerplate -> split into segments -> classify every segment
with the existing model -> combine the evidence into ONE ticket with a primary
department, optional secondary departments and a priority flag.
"""
import re
from collections import defaultdict

import pandas as pd

from src.routing import get_department

# ---- Tunable settings -------------------------------------------------------
SEGMENT_MIN_CONFIDENCE = 0.35   # a sentence needs this confidence to cast a vote
SECONDARY_MIN_SHARE = 0.20      # other model-based departments need 20% of the evidence
AUTO_ROUTE_MIN_SHARE = 0.50     # primary department needs 50% of the evidence to auto-route
RULE_HIT_WEIGHT = 0.20          # each keyword hit counts as 0.2 of a confident sentence
MIN_SEGMENT_WORDS = 5
MAX_SEGMENTS = 60

# Business rules for topics the model was NOT trained on (edit freely).
# overrides_model=True -> the rule decides the primary department.
TOPIC_RULES = [
    {
        "topic": "Loans and credit",
        "department": "Loans and Lending",
        "keywords": ["loan", "mortgage", "disburse", "disbursed", "disbursement",
                     "emi", "overdraft", "credit limit", "interest rate"],
        "overrides_model": True,
        "min_hits": 2,
    },
    {
        "topic": "Fraud or security",
        "department": "Fraud and Security",
        "keywords": ["fraud", "fraudulent", "scam", "phishing", "unauthorised",
                     "unauthorized", "hacked", "identity theft"],
        "overrides_model": True,
        "min_hits": 2,
    },
    {
        "topic": "Formal complaint",
        "department": "Customer Complaints",
        "keywords": ["formal complaint", "complaint", "unacceptable", "dissatisfied"],
        "overrides_model": False,
        "min_hits": 1,
    },
]

ESCALATION_TERMS = [
    "cfpb", "consumer financial protection bureau", "ombudsman", "regulator",
    "comptroller", "occ", "legal action", "lawyer", "attorney", "solicitor",
    "court", "sue", "escalate", "escalating",
]

# ---- Cleaning ---------------------------------------------------------------
SUBJECT_RE = re.compile(r"^\s*(?:subject|re)\s*:\s*(.+)$", re.I | re.M)
HEADER_RE = re.compile(
    r"^\s*(to|from|date|cc|bcc|phone|tel|mobile|email|e-mail|address|sent)\s*:", re.I
)
GREETING_RE = re.compile(r"^\s*(dear|hello|hi|to whom it may concern)\b", re.I)
CLOSING_RE = re.compile(
    r"^\s*(sincerely|regards|kind regards|best regards|warm regards|respectfully|"
    r"yours (sincerely|faithfully|truly)|thank you|thanks)\b",
    re.I,
)
FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z /()-]{1,40}):\s*(.*)$")
NOISE_RE = re.compile(
    r"(\b[\w.+-]+@[\w-]+\.[\w.-]+\b)"          # emails
    r"|(https?://\S+|www\.\S+)"                 # urls
    r"|(\(?\+?\d[\d\s().-]{7,}\d)"              # phone / reference numbers
)


def _pattern(words):
    ordered = sorted(words, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in ordered) + r")\b", re.I
    )


ESCALATION_RE = _pattern(ESCALATION_TERMS)


def clean_letter(text):
    """Return (subject, body) with letter boilerplate removed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    subject = ""
    match = SUBJECT_RE.search(text)
    if match:
        subject = re.sub(r"\[[^\]]*\]", " ", match.group(1))   # drop [reference numbers]
        subject = NOISE_RE.sub(" ", subject)
        subject = re.sub(r"\s+", " ", subject).strip()

    kept = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if re.match(r"\s*(subject|re)\s*:", line, re.I):
            continue
        if HEADER_RE.match(line) or GREETING_RE.match(line):
            continue
        if CLOSING_RE.match(line) and len(line.split()) <= 6:
            break  # everything after the sign-off is signature
        field = FIELD_RE.match(line)
        if field:
            label, value = field.group(1).strip(), field.group(2).strip()
            words = label.split()
            looks_like_label = len(words) <= 5 and all(
                w[0].isupper() for w in words if w[0].isalpha()
            )
            if looks_like_label:
                if len(value.split()) < 6:
                    continue          # metadata such as "Applicant Name: Jane Doe"
                line = value          # keep the sentence, drop the label
        kept.append(line)

    body = NOISE_RE.sub(" ", "\n".join(kept))
    body = re.sub(r"[ \t]+", " ", body)
    return subject, body.strip()


def split_segments(body):
    pieces = []
    for block in re.split(r"\n+", body):
        block = re.sub(r"^(?:[-*\u2022]\s+|\d+[.)]\s+)", "", block).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if len(sentence.split()) >= MIN_SEGMENT_WORDS:
                pieces.append(sentence)
    return pieces[:MAX_SEGMENTS]


def extra_departments():
    """Departments introduced by the business rules (not in intent_routing.json)."""
    return sorted({rule["department"] for rule in TOPIC_RULES})


# ---- Main entry point -------------------------------------------------------
def triage_letter(text, model, settings, config):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text must be a non-empty string.")

    threshold = float(settings["confidence_threshold"])
    subject, body = clean_letter(text)
    cleaned = f"{subject}. {body}" if subject else body

    segments = ([subject] if len(subject.split()) >= 3 else []) + split_segments(body)

    scores = defaultdict(float)
    intent_scores = defaultdict(lambda: defaultdict(float))
    best_conf = defaultdict(float)
    rows = []

    if segments:
        probabilities = model.predict_proba(segments)
        for segment, probs in zip(segments, probabilities):
            index = int(probs.argmax())
            intent = str(model.classes_[index])
            confidence = float(probs[index])
            department = get_department(intent, config)
            counted = confidence >= SEGMENT_MIN_CONFIDENCE
            if counted:
                scores[department] += confidence
                intent_scores[department][intent] += confidence
                best_conf[department] = max(best_conf[department], confidence)
            rows.append(
                {
                    "Segment": segment,
                    "Intent": intent.replace("_", " ").strip().title(),
                    "Confidence": confidence,
                    "Department": department,
                    "Counted": counted,
                }
            )

    # Business rules.
    rule_hits = []
    for rule in TOPIC_RULES:
        hits = len(_pattern(rule["keywords"]).findall(cleaned))
        if hits >= rule["min_hits"]:
            rule_hits.append(
                {
                    "topic": rule["topic"],
                    "department": rule["department"],
                    "hits": hits,
                    "overrides_model": rule["overrides_model"],
                }
            )
            if not rule["overrides_model"]:
                scores[rule["department"]] += min(hits, 5) * RULE_HIT_WEIGHT

    escalations = sorted({m.lower() for m in ESCALATION_RE.findall(text)})
    priority = "High" if escalations else "Normal"

    # Primary department.
    forced = max(
        (r for r in rule_hits if r["overrides_model"]),
        key=lambda r: r["hits"],
        default=None,
    )
    total = sum(scores.values())
    shares = {d: s / total for d, s in scores.items()} if total else {}

    if forced:
        primary, source = forced["department"], "Business rule"
    elif scores:
        primary, source = max(scores, key=scores.get), "Model"
    else:
        primary, source = None, "None"

    # Other departments that should be aware of the ticket.
    secondary = [r["department"] for r in rule_hits if r["department"] != primary]
    for department, share in sorted(shares.items(), key=lambda kv: kv[1], reverse=True):
        if (
            department != primary
            and department not in secondary
            and department != config["default_department"]
            and share >= SECONDARY_MIN_SHARE
            and best_conf.get(department, 0.0) >= threshold
        ):
            secondary.append(department)
    secondary = secondary[:3]

    # Decision.
    proposed = primary or config["default_department"]
    if primary is None:
        decision = "MANUAL_REVIEW"
        reason = "No sentence was confident enough and no business rule matched."
    elif forced:
        decision = "AUTO_ROUTED"
        reason = (
            f"{forced['hits']} mentions of {forced['topic'].lower()} terms matched a "
            "business rule (this topic is outside the model's trained intents)."
        )
    else:
        share = shares.get(primary, 0.0)
        confidence = best_conf.get(primary, 0.0)
        if share >= AUTO_ROUTE_MIN_SHARE and confidence >= threshold:
            decision = "AUTO_ROUTED"
            reason = f"{share:.0%} of the letter's evidence points to {primary}."
        elif share < AUTO_ROUTE_MIN_SHARE:
            decision = "MANUAL_REVIEW"
            reason = f"Evidence is split across departments (primary has {share:.0%})."
        else:
            decision = "MANUAL_REVIEW"
            reason = "No sentence reached the confidence threshold for the primary department."

    intents = intent_scores.get(primary, {}) if primary else {}
    predicted_intent = max(intents, key=intents.get) if intents else ""

    return {
        "mode": "letter",
        "ticket_text": text.strip(),
        "subject": subject,
        "cleaned_text": cleaned,
        "predicted_intent": predicted_intent,
        "confidence": best_conf.get(primary, 0.0) if primary else 0.0,
        "confidence_threshold": threshold,
        "proposed_department": proposed,
        "assigned_department": proposed if decision == "AUTO_ROUTED" else config["manual_review_department"],
        "secondary_departments": secondary,
        "decision": decision,
        "reason": reason,
        "priority": priority,
        "priority_reasons": escalations,
        "primary_source": source,
        "rule_hits": rule_hits,
        "department_shares": shares,
        "segments": pd.DataFrame(rows),
    }