import altair as alt
import pandas as pd
import streamlit as st

from src.letter_triage import TOPIC_RULES
from src.routing import route_ticket
from ui.data import (
    ARTIFACTS, CONFIG, DATA, department_list, load_model, load_train,
    pretty, read_json, require, routing_config,
)
from ui.ledger import AUTO, PENDING, REVIEWED, clear_all, ledger_frame, register_ticket
from ui.theme import AMBER, GREEN, PRIMARY, badge, kpi_row, page_header, section, show_chart, table

require(
    ARTIFACTS / "best_model.joblib", ARTIFACTS / "routing_settings.json",
    CONFIG, DATA / "train.csv", DATA / "test.csv",
)

departments = department_list()
config = routing_config()

CONF_COL = {
    "Confidence": st.column_config.ProgressColumn(
        "Confidence", min_value=0.0, max_value=1.0, format="percent"
    )
}

page_header(
    "Department Workload",
    "See which departments handle tickets the system routed automatically, "
    "which received tickets after human review, and what is still waiting.",
    eyebrow="Workspace",
)

# ------------------------------------------------------- Sample traffic tools
with st.expander("Generate sample traffic or reset data"):
    st.caption(
        "Runs real BANKING77 test messages through the classifier and records them "
        "exactly like tickets submitted on the Ticket Triage page."
    )
    c1, c2, c3 = st.columns([3, 1, 1], gap="medium")
    n_sample = c1.slider("Number of sample tickets", 10, 300, 60, step=10)
    c2.write("")
    c3.write("")
    if c2.button("Simulate", type="primary", icon=":material/play_arrow:", width="stretch"):
        model = load_model()
        settings = read_json("routing_settings.json")
        pool = pd.read_csv(DATA / "test.csv").dropna(subset=["text"])
        sample = pool.sample(min(n_sample, len(pool)))["text"].astype(str).tolist()
        progress = st.progress(0.0, text="Classifying sample tickets...")
        for i, text in enumerate(sample, start=1):
            result = route_ticket(text, model=model, settings=settings, config=config)
            result["mode"] = "short"
            register_ticket(result, source="Simulated")
            progress.progress(i / len(sample), text=f"Classified {i} of {len(sample)}")
        progress.empty()
        st.rerun()
    if c3.button("Clear all", icon=":material/delete:", width="stretch"):
        clear_all()
        st.rerun()

ledger = ledger_frame()
total = len(ledger)
n_auto = int((ledger["Route"] == AUTO).sum())
n_review = int((ledger["Route"] == REVIEWED).sum())
n_wait = int((ledger["Route"] == PENDING).sum())

# ------------------------------------------------------------ Per-department
rows = []
for dept in departments:
    handled_auto = (ledger["Route"] == AUTO) & (ledger["Department"] == dept)
    handled_rev = (ledger["Route"] == REVIEWED) & (ledger["Department"] == dept)
    waiting = (ledger["Route"] == PENDING) & (ledger["Suggested department"] == dept)
    copied = ledger["Also involves"].fillna("").str.contains(dept, regex=False)
    rows.append(
        {
            "Department": dept,
            "Auto-routed": int(handled_auto.sum()),
            "Routed after review": int(handled_rev.sum()),
            "Awaiting review (suggested)": int(waiting.sum()),
            "Also copied on": int(copied.sum()),
            "High priority": int(
                ((handled_auto | handled_rev | waiting) & (ledger["Priority"] == "High")).sum()
            ),
        }
    )
summary = pd.DataFrame(rows)
summary["Handled"] = summary["Auto-routed"] + summary["Routed after review"]
summary["Total"] = summary["Handled"] + summary["Awaiting review (suggested)"]
active = int((summary["Total"] > 0).sum())

kpi_row(
    [
        {"label": "Tickets classified", "value": f"{total:,}", "tone": "indigo"},
        {"label": "Auto-routed", "value": f"{n_auto:,}",
         "sub": f"{n_auto / total:.0%} of tickets" if total else "No tickets yet", "tone": "green"},
        {"label": "Routed after review", "value": f"{n_review:,}",
         "sub": "Approved by a reviewer", "tone": "slate"},
        {"label": "In manual review queue", "value": f"{n_wait:,}",
         "sub": "Waiting for a reviewer", "tone": "amber"},
        {"label": "Active departments", "value": f"{active} / {len(departments)}", "tone": "indigo"},
    ]
)
st.write("")

tab_load, tab_dir, tab_ledger = st.tabs(
    ["Workload by department", "Department directory", "Ticket ledger"]
)

# ---------------------------------------------------------------- Workload
with tab_load:
    if total == 0:
        st.info(
            "No tickets have been classified in this session yet. Submit tickets on the "
            "Ticket Triage page, or open “Generate sample traffic” above."
        )
    else:
        section(
            "Who is handling what",
            "Green: sent straight to the department. Indigo: sent after a reviewer approved it. "
            "Amber: waiting in the manual review queue (counted under the suggested department).",
        )
        shown = summary[summary["Total"] > 0].sort_values("Total", ascending=False)
        long = shown.melt(
            id_vars="Department",
            value_vars=["Auto-routed", "Routed after review", "Awaiting review (suggested)"],
            var_name="Route", value_name="Tickets",
        )
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("Department:N", sort=shown["Department"].tolist(), title=None,
                        axis=alt.Axis(labelLimit=240)),
                x=alt.X("Tickets:Q", title="Tickets", stack="zero"),
                color=alt.Color(
                    "Route:N",
                    scale=alt.Scale(
                        domain=["Auto-routed", "Routed after review", "Awaiting review (suggested)"],
                        range=[GREEN, PRIMARY, AMBER],
                    ),
                ),
                tooltip=["Department", "Route", "Tickets"],
            )
            .properties(height=max(220, 44 * len(shown)))
        )
        show_chart(chart)

        section("Department summary")
        table(
            shown[["Department", "Auto-routed", "Routed after review",
                   "Awaiting review (suggested)", "Handled", "Also copied on", "High priority"]]
        )

        section("Department drill-down", "Inspect the tickets behind the numbers.")
        choice = st.selectbox("Department", departments)
        is_waiting = (ledger["Route"] == PENDING) & (ledger["Suggested department"] == choice)
        is_handled = (ledger["Route"] != PENDING) & (ledger["Department"] == choice)
        detail = ledger[is_waiting | is_handled]

        kpi_row(
            [
                {"label": "Auto-routed here", "value": int((detail["Route"] == AUTO).sum()), "tone": "green"},
                {"label": "Routed after review", "value": int((detail["Route"] == REVIEWED).sum()), "tone": "indigo"},
                {"label": "Awaiting review", "value": int((detail["Route"] == PENDING).sum()), "tone": "amber"},
                {"label": "High priority", "value": int((detail["Priority"] == "High").sum()), "tone": "red"},
            ]
        )
        st.write("")
        if detail.empty:
            st.info(f"No tickets have been routed to {choice} yet.")
        else:
            table(
                detail[["Time", "ID", "Type", "Message", "Intent", "Confidence", "Route", "Priority"]],
                column_config=CONF_COL,
            )

# --------------------------------------------------------------- Directory
with tab_dir:
    section(
        "Who handles what",
        "Each department, the intents routed to it by `config/intent_routing.json`, "
        "and its current workload.",
    )
    train = load_train()
    dept_intents = {
        dept: sorted(group["label"].unique()) for dept, group in train.groupby("department")
    }
    rule_by_dept = {rule["department"]: rule for rule in TOPIC_RULES}
    queue_name = config["manual_review_department"]

    cols = st.columns(3, gap="medium")
    for i, dept in enumerate(departments):
        row = summary[summary["Department"] == dept].iloc[0]
        intents = dept_intents.get(dept, [])
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{dept}**")
                chips = badge(f"{int(row['Handled'])} handled", "ok")
                if row["Awaiting review (suggested)"]:
                    chips += badge(f"{int(row['Awaiting review (suggested)'])} awaiting", "warn")
                if intents:
                    chips += badge(f"{len(intents)} intents", "info")
                st.markdown(chips, unsafe_allow_html=True)

                if intents:
                    preview = ", ".join(pretty(x) for x in intents[:4])
                    more = f" +{len(intents) - 4} more" if len(intents) > 4 else ""
                    st.caption(preview + more)
                    with st.expander("View all intents"):
                        st.write(", ".join(pretty(x) for x in intents))
                elif dept in rule_by_dept:
                    rule = rule_by_dept[dept]
                    st.caption(
                        "Handled by business rule: topics such as "
                        + ", ".join(rule["keywords"][:6]) + "."
                    )
                else:
                    st.caption("No intents currently map to this department.")

    st.info(
        f"Tickets the system is not confident about are held in **{queue_name}** until a "
        "reviewer assigns a department. Letters can also copy additional departments "
        "(shown as “Also copied on”)."
    )

# ------------------------------------------------------------------ Ledger
with tab_ledger:
    section("Ticket ledger", "Every ticket classified in this session, newest first.")
    if total == 0:
        st.info("The ledger is empty.")
    else:
        f1, f2, f3, f4 = st.columns([2, 2, 1, 2])
        routes = f1.multiselect("Route", [AUTO, REVIEWED, PENDING], default=[AUTO, REVIEWED, PENDING])
        depts = f2.multiselect("Department", departments)
        prio = f3.selectbox("Priority", ["All", "High", "Normal"])
        text_q = f4.text_input("Search message")

        view = ledger[ledger["Route"].isin(routes)]
        if depts:
            view = view[
                view["Department"].isin(depts) | view["Suggested department"].isin(depts)
            ]
        if prio != "All":
            view = view[view["Priority"] == prio]
        if text_q.strip():
            view = view[view["Message"].str.contains(text_q.strip(), case=False, regex=False)]

        view = view.iloc[::-1]
        st.caption(f"{len(view):,} of {total:,} tickets")
        table(view, column_config=CONF_COL)
        st.download_button(
            "Download ledger (CSV)", view.to_csv(index=False).encode("utf-8"),
            file_name="ticket_ledger.csv", mime="text/csv", icon=":material/download:",
        )

st.caption(
    "The ledger is stored in the current browser session. It resets when the session ends "
    "and is not shared between users."
)