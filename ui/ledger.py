"""Session-wide register of every ticket the system has classified."""
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

AUTO = "Auto-routed"
PENDING = "Awaiting review"
REVIEWED = "Routed after review"

COLUMNS = [
    "ID", "Time", "Source", "Type", "Message", "Intent", "Confidence", "Route",
    "Department", "Suggested department", "Also involves", "Priority",
]


def _preview(text, n=160):
    flat = re.sub(r"\s+", " ", str(text)).strip()
    return flat if len(flat) <= n else flat[:n].rstrip() + "…"


def _intent(value):
    return str(value).replace("_", " ").strip().title() if value else "Outside trained intents"


def register_ticket(result, queue_manual=True, source="Entered"):
    """Record a classified ticket. Manual-review tickets also join the review queue."""
    st.session_state.setdefault("tickets", [])
    st.session_state.setdefault("review_queue", [])

    ticket_id = uuid.uuid4().hex[:8]
    now = datetime.now()
    auto = result["decision"] == "AUTO_ROUTED"

    st.session_state.tickets.append(
        {
            "ticket_id": ticket_id,
            "created": now.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "type": "Letter" if result.get("mode") == "letter" else "Short",
            "message": result["ticket_text"],
            "intent": result.get("predicted_intent", ""),
            "confidence": result.get("confidence", 0.0),
            "route": AUTO if auto else PENDING,
            "suggested_department": result["proposed_department"],
            "department": result["assigned_department"],
            "secondary": ", ".join(result.get("secondary_departments", [])),
            "priority": result.get("priority", "Normal"),
        }
    )

    if not auto and queue_manual:
        st.session_state.review_queue.append(
            {
                "ticket_id": ticket_id,
                "created": now.strftime("%H:%M:%S"),
                "ticket_text": result["ticket_text"],
                "predicted_intent": result.get("predicted_intent", ""),
                "confidence": result.get("confidence", 0.0),
                "proposed_department": result["proposed_department"],
                "secondary_departments": result.get("secondary_departments", []),
                "priority": result.get("priority", "Normal"),
                "assigned_department": "",
                "status": "Pending Review",
            }
        )
    return ticket_id


def mark_routed(ticket_id, department):
    """A reviewer approved a queued ticket and assigned it to a department."""
    for ticket in st.session_state.get("tickets", []):
        if ticket["ticket_id"] == ticket_id:
            ticket["route"] = REVIEWED
            ticket["department"] = department
            break


def clear_all():
    st.session_state.tickets = []
    st.session_state.review_queue = []
    st.session_state.history = []


def ledger_frame():
    rows = [
        {
            "ID": t["ticket_id"],
            "Time": t["created"],
            "Source": t["source"],
            "Type": t["type"],
            "Message": _preview(t["message"]),
            "Intent": _intent(t["intent"]),
            "Confidence": t["confidence"],
            "Route": t["route"],
            "Department": t["department"],
            "Suggested department": t["suggested_department"],
            "Also involves": t["secondary"],
            "Priority": t["priority"],
        }
        for t in st.session_state.get("tickets", [])
    ]
    return pd.DataFrame(rows, columns=COLUMNS)