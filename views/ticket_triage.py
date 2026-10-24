import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ui.ledger import mark_routed, register_ticket
from src.letter_triage import triage_letter
from src.routing import route_ticket
from ui.data import (
    ARTIFACTS, CONFIG, department_list, load_model, pretty,
    read_json, require, routing_config,
)
from ui.theme import (
    badge, confidence_meter, decision_banner, hbar, kpi_row,
    page_header, quote, section, show_chart, table,
)

require(ARTIFACTS / "best_model.joblib", ARTIFACTS / "routing_settings.json", CONFIG)

model = load_model()
settings = read_json("routing_settings.json")
config = routing_config()
departments = department_list()

LETTER_WORD_LIMIT = 40
OUTSIDE = "Outside trained intents"

SAMPLE_LETTER = """Subject: Formal complaint - approved loan not disbursed

Dear Sir or Madam,

My personal loan was approved on 22 September and I was told the money would reach my account within 21 business days. It is now more than 30 days and the loan has still not been disbursed. I have called customer support twice and received conflicting answers. This is unacceptable. Please disburse the loan immediately and explain the delay, otherwise I will escalate this to the banking ombudsman.

Sincerely,
Jane Doe"""

EXAMPLES = [
    ("Transfer not arrived", "My bank transfer has not arrived yet."),
    ("Card still missing", "I am still waiting for my new bank card."),
    ("Unexpected fee", "Why was I charged an extra fee on my statement?"),
    ("Vague request", "I need help with something in my account."),
    ("Complaint letter", SAMPLE_LETTER),
]

CONF_COL = {
    "Confidence": st.column_config.ProgressColumn(
        "Confidence", min_value=0.0, max_value=1.0, format="percent"
    )
}

for key, default in {
    "review_queue": [], "history": [], "ticket_input": "",
    "last_result": None, "last_top": None, "batch_result": None,
}.items():
    st.session_state.setdefault(key, default)


def use_example(text):
    st.session_state.ticket_input = text


def now():
    return datetime.now().strftime("%H:%M:%S")


def preview(text, n=160):
    flat = re.sub(r"\s+", " ", str(text)).strip()
    return flat if len(flat) <= n else flat[:n].rstrip() + "…"


def intent_label(intent):
    return pretty(intent) if intent else OUTSIDE


def conf_text(value):
    return f"{value:.0%}" if value else "n/a"


def run_triage(text, mode="Auto-detect"):
    """Short messages use the standard path; long text is handled as a letter."""
    as_letter = mode == "Full letter" or (
        mode == "Auto-detect" and len(text.split()) > LETTER_WORD_LIMIT
    )
    if as_letter:
        return triage_letter(text, model, settings, config)
    result = route_ticket(text, model=model, settings=settings, config=config)
    result["mode"] = "short"
    return result


def top_predictions(text, n=5):
    probs = model.predict_proba([text])[0]
    idx = np.argsort(probs)[::-1][:n]
    return pd.DataFrame(
        {"intent": [pretty(model.classes_[i]) for i in idx],
         "probability": [float(probs[i]) for i in idx]}
    )


page_header(
    "Ticket Triage",
    "Classify short messages and full complaint letters. Each submission becomes "
    "one ticket; confident ones are routed automatically, the rest go to a reviewer.",
    eyebrow="Workspace",
)

queue = st.session_state.review_queue
pending = sorted(
    [t for t in queue if t["status"] == "Pending Review"],
    key=lambda t: t.get("priority") != "High",   # high priority first
)
resolved = [t for t in queue if t["status"] != "Pending Review"]

tab_single, tab_batch, tab_queue, tab_log = st.tabs(
    ["Single ticket", "Batch triage", f"Review queue ({len(pending)})", "Activity log"]
)

# ------------------------------------------------------------------ Single
with tab_single:
    st.text_area(
        "Customer message or full letter",
        key="ticket_input",
        height=220,
        placeholder="Paste a short message or an entire complaint letter. "
                    "The whole text is treated as ONE ticket.",
    )

    mode = st.radio(
        "Analysis mode",
        ["Auto-detect", "Short message", "Full letter"],
        horizontal=True,
        help=f"Auto-detect uses letter mode above {LETTER_WORD_LIMIT} words.",
    )

    st.caption("Try an example")
    for col, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(label, key=f"ex_{label}", on_click=use_example, args=(text,), width="stretch")

    if st.button("Classify & route ticket", type="primary", icon=":material/bolt:"):
        text = st.session_state.ticket_input
        if not text.strip():
            st.warning("Please enter a support message first.")
        else:
            try:
                with st.spinner("Analyzing ticket..."):
                    result = run_triage(text, mode)
                    top = top_predictions(text) if result["mode"] == "short" else None
                st.session_state.last_result = result
                st.session_state.last_top = top
                st.session_state.history.insert(
                    0,
                    {
                        "Time": now(),
                        "Type": "Letter" if result["mode"] == "letter" else "Short",
                        "Message": preview(result["ticket_text"]),
                        "Intent": intent_label(result["predicted_intent"]),
                        "Confidence": result["confidence"],
                        "Decision": result["decision"].replace("_", " ").title(),
                        "Department": result["assigned_department"],
                    },
                )
                register_ticket(result)
            except Exception as error:
                st.error("The ticket could not be processed.")
                st.exception(error)

    result = st.session_state.last_result
    if result:
        auto = result["decision"] == "AUTO_ROUTED"
        is_letter = result.get("mode") == "letter"
        section("Result")

        if auto:
            decision_banner(True, f"Auto-routed to {result['assigned_department']}", result["reason"])
        else:
            decision_banner(
                False,
                "Sent to manual review",
                f"{result['reason']} Suggested department: {result['proposed_department']}.",
            )

        if result["confidence"]:
            conf_value = f"{result['confidence']:.1%}"
        else:
            conf_value = "Rule-based" if result.get("primary_source") == "Business rule" else "n/a"

        kpi_row(
            [
                {"label": "Primary intent" if is_letter else "Predicted intent",
                 "value": intent_label(result["predicted_intent"]), "tone": "indigo"},
                {"label": "Confidence", "value": conf_value, "tone": "green" if auto else "amber"},
                {"label": "Primary department", "value": result["proposed_department"], "tone": "slate"},
            ]
        )

        if is_letter:
            chips = badge(
                f"Priority: {result['priority']}",
                "warn" if result["priority"] == "High" else "muted",
            )
            for dept in result["secondary_departments"]:
                chips += badge(f"Also involves: {dept}", "info")
            st.markdown(f"<div style='margin-top:.8rem'>{chips}</div>", unsafe_allow_html=True)
            if result["priority_reasons"]:
                st.caption("Escalation language detected: " + ", ".join(result["priority_reasons"]))

            left, right = st.columns(2, gap="large")
            with left:
                section("Model evidence by department", "Share of confident sentences per department.")
                shares = result["department_shares"]
                if shares:
                    share_df = pd.DataFrame(
                        {"department": list(shares.keys()), "share": list(shares.values())}
                    )
                    show_chart(hbar(share_df, "share", "department", fmt=".0%", height=200))
                else:
                    st.info("No sentence was confident enough to vote.")
            with right:
                section("Business rules triggered")
                if result["rule_hits"]:
                    table(
                        pd.DataFrame(
                            [
                                {
                                    "Topic": r["topic"],
                                    "Department": r["department"],
                                    "Mentions": r["hits"],
                                    "Decides routing": "Yes" if r["overrides_model"] else "No",
                                }
                                for r in result["rule_hits"]
                            ]
                        )
                    )
                else:
                    st.caption("No business rules matched this letter.")

            with st.expander("Evidence: sentences the model classified"):
                segments = result["segments"]
                if segments.empty:
                    st.write("No usable sentences were found after cleaning the letter.")
                else:
                    table(segments, column_config=CONF_COL)
                    st.caption(
                        "Only sentences marked “Counted” (≥ 35% confidence) vote. "
                        "Header, greeting, signature, contact details and field labels "
                        "are removed before classification."
                    )
            with st.expander("Text used after removing letter boilerplate"):
                st.write(result["cleaned_text"] or "(empty)")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                section("Confidence vs. threshold")
                confidence_meter(result["confidence"], result["confidence_threshold"])
                st.write("")
                st.markdown(quote(result["ticket_text"]), unsafe_allow_html=True)
            with right:
                section("Top 5 candidate intents")
                if st.session_state.last_top is not None:
                    show_chart(
                        hbar(st.session_state.last_top, "probability", "intent", fmt=".1%", height=200)
                    )

# ------------------------------------------------------------------- Batch
with tab_batch:
    section(
        "Batch triage",
        "Upload a CSV with a `text` column (one ticket per row), or paste text. "
        "Long entries are automatically handled as letters.",
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    pasted = st.text_area("…or paste text", height=140, key="batch_text")
    split_mode = st.radio(
        "How should pasted text be split into tickets?",
        ["One ticket per line", "Tickets separated by a --- line", "Whole text is one ticket"],
        horizontal=True,
    )
    add_to_queue = st.checkbox("Add low-confidence tickets to the review queue", value=True)

    if st.button("Run batch triage", type="primary", icon=":material/playlist_play:"):
        texts = []
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                column = "text" if "text" in frame.columns else frame.columns[0]
                texts = frame[column].dropna().astype(str).tolist()
            except Exception as error:
                st.error("The CSV file could not be read.")
                st.exception(error)
        elif pasted.strip():
            if split_mode == "One ticket per line":
                texts = [line.strip() for line in pasted.splitlines() if line.strip()]
            elif split_mode.startswith("Tickets separated"):
                texts = [
                    block.strip()
                    for block in re.split(r"^\s*-{3,}\s*$", pasted, flags=re.M)
                    if block.strip()
                ]
            else:
                texts = [pasted.strip()]

        texts = texts[:500]
        if not texts:
            st.warning("Provide a CSV file or paste some text.")
        else:
            rows = []
            progress = st.progress(0.0, text="Classifying tickets...")
            for i, text in enumerate(texts, start=1):
                try:
                    res = run_triage(text)
                except Exception:
                    progress.progress(i / len(texts))
                    continue
                register_ticket(res, queue_manual=add_to_queue)
                rows.append(
                    {
                        "Type": "Letter" if res["mode"] == "letter" else "Short",
                        "Message": preview(res["ticket_text"]),
                        "Intent": intent_label(res["predicted_intent"]),
                        "Confidence": res["confidence"],
                        "Decision": res["decision"].replace("_", " ").title(),
                        "Department": res["assigned_department"],
                        "Also involves": ", ".join(res.get("secondary_departments", [])),
                        "Priority": res.get("priority", "Normal"),
                    }
                )
                progress.progress(i / len(texts), text=f"Classified {i} of {len(texts)}")
            progress.empty()
            st.session_state.batch_result = pd.DataFrame(rows)
            st.rerun()

    batch = st.session_state.batch_result
    if batch is not None and not batch.empty:
        auto_count = int((batch["Decision"] == "Auto Routed").sum())
        kpi_row(
            [
                {"label": "Tickets processed", "value": f"{len(batch):,}", "tone": "indigo"},
                {"label": "Auto-routed", "value": f"{auto_count:,}",
                 "sub": f"{auto_count / len(batch):.0%}", "tone": "green"},
                {"label": "Manual review", "value": f"{len(batch) - auto_count:,}",
                 "sub": f"{(len(batch) - auto_count) / len(batch):.0%}", "tone": "amber"},
                {"label": "High priority", "value": f"{int((batch['Priority'] == 'High').sum()):,}",
                 "tone": "red"},
            ]
        )
        st.write("")
        table(batch, column_config=CONF_COL)
        st.download_button(
            "Download results (CSV)",
            batch.to_csv(index=False).encode("utf-8"),
            file_name="batch_triage_results.csv",
            mime="text/csv",
            icon=":material/download:",
        )

# ------------------------------------------------------------------- Queue
with tab_queue:
    kpi_row(
        [
            {"label": "Pending review", "value": len(pending), "tone": "amber"},
            {"label": "High priority", "value": sum(1 for t in pending if t.get("priority") == "High"),
             "tone": "red"},
            {"label": "Resolved", "value": len(resolved), "tone": "green"},
            {"label": "Total in queue", "value": len(queue), "tone": "slate"},
        ]
    )

    if not queue:
        st.info("The queue is empty. Submit a low-confidence ticket (try the “Vague request” example).")

    for ticket in pending:
        with st.container(border=True):
            info, action = st.columns([3, 2], gap="large")
            with info:
                chips = badge("Pending review", "warn")
                if ticket.get("priority") == "High":
                    chips += badge("High priority", "warn")
                chips += badge(intent_label(ticket["predicted_intent"]), "info")
                chips += badge(f"{conf_text(ticket['confidence'])} confidence", "muted")
                chips += badge(f"#{ticket['ticket_id']}", "muted")
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown(quote(preview(ticket["ticket_text"], 300)), unsafe_allow_html=True)
                if len(ticket["ticket_text"]) > 300:
                    with st.expander("Read full message"):
                        st.text(ticket["ticket_text"])
                also = ticket.get("secondary_departments", [])
                st.caption(
                    f"Created {ticket['created']} · Suggested: {ticket['proposed_department']}"
                    + (f" · Also involves: {', '.join(also)}" if also else "")
                )
            with action:
                default_index = (
                    departments.index(ticket["proposed_department"])
                    if ticket["proposed_department"] in departments else 0
                )
                chosen = st.selectbox(
                    "Assign department", departments, index=default_index,
                    key=f"dept_{ticket['ticket_id']}",
                )
                if st.button("Approve & route", key=f"approve_{ticket['ticket_id']}",
                             type="primary", width="stretch"):
                    ticket["status"] = "Routed"
                    ticket["assigned_department"] = chosen
                    mark_routed(ticket["ticket_id"], chosen)
                    st.rerun()

    if resolved:
        section("Resolved tickets")
        done = pd.DataFrame(resolved)[
            ["ticket_id", "ticket_text", "predicted_intent", "confidence",
             "assigned_department", "priority", "status"]
        ].rename(
            columns={
                "ticket_id": "ID", "ticket_text": "Message", "predicted_intent": "Intent",
                "confidence": "Confidence", "assigned_department": "Department",
                "priority": "Priority", "status": "Status",
            }
        )
        done["Message"] = done["Message"].map(preview)
        done["Intent"] = done["Intent"].map(intent_label)
        table(done, column_config=CONF_COL)
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download resolved tickets (CSV)", done.to_csv(index=False).encode("utf-8"),
            file_name="resolved_tickets.csv", mime="text/csv", icon=":material/download:",
        )
        if c2.button("Clear resolved tickets", icon=":material/delete:"):
            st.session_state.review_queue = [
                t for t in st.session_state.review_queue if t["status"] == "Pending Review"
            ]
            st.rerun()

# --------------------------------------------------------------------- Log
with tab_log:
    history = st.session_state.history
    if not history:
        st.info("No tickets classified yet in this session.")
    else:
        table(pd.DataFrame(history), column_config=CONF_COL)
        if st.button("Clear activity log", icon=":material/delete:"):
            st.session_state.history = []
            st.rerun()