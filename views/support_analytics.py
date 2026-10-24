import pandas as pd
import streamlit as st

from ui.data import DATA, load_train, pretty, require
from ui.theme import PRIMARY, hbar, kpi_row, page_header, section, show_chart, table

require(DATA / "train.csv")

df = load_train()
total = len(df)
intent_counts = df["label"].value_counts()

page_header(
    "Support Analytics",
    "Explore the distribution of customer issues in the BANKING77 training "
    "dataset. These are dataset statistics, not live operational metrics.",
    eyebrow="Insights",
)

kpi_row(
    [
        {"label": "Dataset tickets", "value": f"{total:,}", "tone": "indigo"},
        {"label": "Support intents", "value": df["label"].nunique(), "tone": "green"},
        {"label": "Departments", "value": df["department"].nunique(), "tone": "amber"},
        {"label": "Avg. message length", "value": f"{df['words'].mean():.1f} words", "tone": "slate"},
    ]
)

tab_dist, tab_explore, tab_search, tab_table = st.tabs(
    ["Distribution", "Intent explorer", "Search messages", "Data table"]
)

# ------------------------------------------------------------ Distribution
with tab_dist:
    left, right = st.columns([3, 2], gap="large")

    with left:
        section("Issue distribution")
        c1, c2 = st.columns([3, 2])
        top_n = c1.slider("Categories to display", 5, 30, 12)
        order = c2.radio("Order", ["Most frequent", "Least frequent"], horizontal=True)
        counts = intent_counts if order == "Most frequent" else intent_counts.sort_values()
        chart_df = pd.DataFrame(
            {"intent": [pretty(i) for i in counts.head(top_n).index],
             "tickets": counts.head(top_n).values.astype(int)}
        )
        show_chart(hbar(chart_df, "tickets", "intent"))

    with right:
        section("Volume by department", "Intents mapped through your routing rules.")
        dept = df["department"].value_counts().reset_index()
        dept.columns = ["department", "tickets"]
        show_chart(hbar(dept, "tickets", "department", color="#0EA5E9"))
        st.caption(
            "Departments come from `config/intent_routing.json`. Rules are matched "
            "in order, so earlier rules take priority."
        )

# ---------------------------------------------------------------- Explorer
with tab_explore:
    section("Intent explorer")
    selected = st.selectbox(
        "Choose a support issue category",
        sorted(intent_counts.index),
        format_func=pretty,
    )
    rows = df[df["label"] == selected]
    kpi_row(
        [
            {"label": "Examples", "value": len(rows), "tone": "indigo"},
            {"label": "Share of dataset", "value": f"{len(rows) / total:.2%}", "tone": "green"},
            {"label": "Department", "value": rows["department"].iloc[0], "tone": "amber"},
            {"label": "Avg. words", "value": f"{rows['words'].mean():.1f}", "tone": "slate"},
        ]
    )
    st.write("")
    n_examples = st.slider("Example messages to show", 5, min(50, len(rows)), min(10, len(rows)))
    table(rows[["text"]].head(n_examples).rename(columns={"text": "Customer message"}))

# ------------------------------------------------------------------ Search
with tab_search:
    section("Search messages", "Find example tickets by keyword.")
    c1, c2 = st.columns([3, 2])
    query = c1.text_input("Keyword", placeholder="e.g. refund, pin, exchange rate")
    depts = c2.multiselect("Department", sorted(df["department"].unique()))

    found = df
    if depts:
        found = found[found["department"].isin(depts)]
    if query.strip():
        found = found[found["text"].str.contains(query.strip(), case=False, regex=False, na=False)]

    st.caption(f"{len(found):,} matching messages")
    if query.strip() or depts:
        view = found[["text", "label", "department"]].rename(
            columns={"text": "Customer message", "label": "Intent", "department": "Department"}
        )
        view["Intent"] = view["Intent"].map(pretty)
        table(view.head(200))
        st.download_button(
            "Download matches (CSV)", view.to_csv(index=False).encode("utf-8"),
            file_name="search_results.csv", mime="text/csv", icon=":material/download:",
        )
    else:
        st.info("Enter a keyword or choose a department to see matching messages.")

# ------------------------------------------------------------------- Table
with tab_table:
    section("Intent frequency table")
    freq = (
        df.groupby(["label", "department"]).size().reset_index(name="Tickets")
        .sort_values("Tickets", ascending=False)
    )
    freq["Share"] = freq["Tickets"] / total
    freq["label"] = freq["label"].map(pretty)
    freq = freq.rename(columns={"label": "Intent", "department": "Department"})
    table(
        freq,
        column_config={
            "Share": st.column_config.ProgressColumn(
                "Share of dataset", min_value=0.0, max_value=float(freq["Share"].max()), format="percent"
            )
        },
    )
    st.download_button(
        "Download table (CSV)", freq.to_csv(index=False).encode("utf-8"),
        file_name="intent_frequency.csv", mime="text/csv", icon=":material/download:",
    )

st.info(
    "Interpretation: frequent categories describe the benchmark training set. "
    "They do not show which issues are most common in a real bank's support operation."
)