import altair as alt
import pandas as pd
import streamlit as st

from ui.data import ARTIFACTS, pretty, read_artifact_csv, read_json, require
from ui.theme import (
    AMBER, GREEN, PRIMARY, hbar, kpi_row, page_header, section, show_chart, table,
)

require(
    ARTIFACTS / "evaluation_summary.json",
    ARTIFACTS / "per_class_metrics.csv",
    ARTIFACTS / "test_predictions.csv",
)

summary = read_json("evaluation_summary.json")
metrics = read_artifact_csv("per_class_metrics.csv")
preds = read_artifact_csv("test_predictions.csv")

# per_class_metrics.csv also stores aggregate rows; keep real intents only.
metrics = metrics[~metrics["intent"].isin(["accuracy", "macro avg", "weighted avg"])].copy()

preds = preds.copy()
errors = preds[preds["label"] != preds["predicted_label"]].copy()
error_rate = len(errors) / len(preds) if len(preds) else 0

page_header(
    "Model Evaluation",
    "Held-out test results for the support intent classifier. These are "
    "benchmark results, not guarantees on real customer tickets.",
    eyebrow="Insights",
)

kpi_row(
    [
        {"label": "Accuracy", "value": f"{summary['accuracy']:.2%}", "tone": "indigo"},
        {"label": "Macro-F1", "value": f"{summary['macro_f1']:.3f}", "tone": "green"},
        {"label": "Weighted-F1", "value": f"{summary['weighted_f1']:.3f}", "tone": "green"},
        {"label": "Test examples", "value": f"{summary['test_examples']:,}", "tone": "slate"},
        {"label": "Errors", "value": f"{len(errors):,}", "sub": f"{error_rate:.2%} error rate", "tone": "red"},
    ]
)

tab_perf, tab_thr, tab_conf, tab_err = st.tabs(
    ["Performance", "Threshold analysis", "Confusion", "Error explorer"]
)

# ------------------------------------------------------------- Performance
with tab_perf:
    left, right = st.columns([3, 2], gap="large")

    with left:
        section("Performance by intent")
        c1, c2 = st.columns([3, 2])
        n = c1.slider("Intents to show", 5, 30, 12)
        mode = c2.radio("Show", ["Lowest F1", "Highest F1"], horizontal=True)
        ranked = metrics.sort_values("f1-score", ascending=(mode == "Lowest F1")).head(n)
        chart_df = pd.DataFrame(
            {"intent": ranked["intent"].map(pretty), "f1": ranked["f1-score"].astype(float)}
        )
        show_chart(
            hbar(chart_df, "f1", "intent", color=AMBER if mode == "Lowest F1" else GREEN,
                 domain=(0, 1), fmt=".2f")
        )

    with right:
        section("Intent drill-down")
        intent = st.selectbox("Select intent", sorted(metrics["intent"]), format_func=pretty)
        row = metrics[metrics["intent"] == intent].iloc[0]
        kpi_row(
            [
                {"label": "Precision", "value": f"{row['precision']:.3f}", "tone": "indigo"},
                {"label": "Recall", "value": f"{row['recall']:.3f}", "tone": "green"},
                {"label": "F1", "value": f"{row['f1-score']:.3f}", "tone": "amber"},
            ]
        )
        st.write("")
        wrong = errors[errors["label"] == intent]["predicted_label"].value_counts()
        st.markdown("**Most often mistaken for**")
        if wrong.empty:
            st.success("No misclassified test examples for this intent.")
        else:
            table(
                pd.DataFrame({"Predicted as": wrong.index.map(pretty), "Count": wrong.values}).head(8)
            )

    section("All intents")
    full = metrics[["intent", "precision", "recall", "f1-score", "support"]].copy()
    full["intent"] = full["intent"].map(pretty)
    table(
        full.sort_values("f1-score"),
        column_config={
            "f1-score": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0, format="%.3f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "support": st.column_config.NumberColumn("Support", format="%d"),
        },
    )

    if (ARTIFACTS / "validation_results.csv").exists():
        section("Model comparison (validation set)", "Algorithms compared before the final model was chosen.")
        val = read_artifact_csv("validation_results.csv")
        long = val.melt(id_vars="model", var_name="metric", value_name="score")
        long["model"] = long["model"].map(pretty)
        long["metric"] = long["metric"].map(pretty)
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=alt.Scale(range=[PRIMARY, GREEN])),
                xOffset="metric:N",
                tooltip=["model", "metric", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(height=260)
        )
        show_chart(chart)

# --------------------------------------------------------------- Threshold
with tab_thr:
    section(
        "Confidence threshold explorer",
        "See how the threshold trades automation against accuracy (validation split).",
    )
    if not (ARTIFACTS / "threshold_analysis.csv").exists():
        st.info("`threshold_analysis.csv` was not found. Run `python src/calibrate.py`.")
    else:
        ta = read_artifact_csv("threshold_analysis.csv").dropna(subset=["auto_route_accuracy"])
        options = [round(float(t), 2) for t in ta["threshold"]]
        current = None
        if (ARTIFACTS / "routing_settings.json").exists():
            current = float(read_json("routing_settings.json")["confidence_threshold"])
        default = min(options, key=lambda t: abs(t - current)) if current is not None else options[0]

        chosen = st.select_slider("Confidence threshold", options=options, value=default, format_func=lambda t: f"{t:.0%}")
        pick = ta.iloc[[abs(o - chosen) < 1e-9 for o in options].index(True)]

        kpi_row(
            [
                {"label": "Auto-routed", "value": f"{pick['coverage']:.1%}",
                 "sub": f"{int(pick['auto_routed_count']):,} tickets", "tone": "indigo"},
                {"label": "Manual review", "value": f"{pick['review_rate']:.1%}", "tone": "amber"},
                {"label": "Auto-route accuracy", "value": f"{pick['auto_route_accuracy']:.1%}", "tone": "green"},
                {"label": "Configured threshold", "value": f"{current:.0%}" if current is not None else "n/a", "tone": "slate"},
            ]
        )
        st.write("")

        long = ta.melt(
            id_vars="threshold",
            value_vars=["coverage", "review_rate", "auto_route_accuracy"],
            var_name="metric", value_name="value",
        )
        long["metric"] = long["metric"].map(
            {"coverage": "Auto-routed share", "review_rate": "Review rate", "auto_route_accuracy": "Auto-route accuracy"}
        )
        lines = (
            alt.Chart(long)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("threshold:Q", title="Confidence threshold", axis=alt.Axis(format="%")),
                y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=alt.Scale(range=[PRIMARY, AMBER, GREEN])),
                tooltip=["metric", alt.Tooltip("threshold:Q", format=".0%"), alt.Tooltip("value:Q", format=".1%")],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"t": [chosen]}))
            .mark_rule(color="#111827", strokeDash=[4, 4])
            .encode(x="t:Q")
        )
        show_chart((lines + rule).properties(height=340))

        with st.expander("View threshold table"):
            table(ta)

# --------------------------------------------------------------- Confusion
with tab_conf:
    section("Most confused intent pairs", "Where the model mixes up one intent for another.")
    if not (ARTIFACTS / "confusion_matrix.csv").exists():
        st.info("`confusion_matrix.csv` was not found. Run `python src/evaluate.py`.")
    else:
        cm = read_artifact_csv("confusion_matrix.csv", index_col=0)
        pairs = cm.stack().reset_index()
        pairs.columns = ["actual", "predicted", "count"]
        pairs = pairs[(pairs["actual"] != pairs["predicted"]) & (pairs["count"] > 0)]
        pairs = pairs.sort_values("count", ascending=False)

        top_k = st.slider("Pairs to show", 5, 25, 12, key="pairs_k")
        top = pairs.head(top_k).copy()
        top["pair"] = top["actual"].map(pretty) + "  →  " + top["predicted"].map(pretty)
        show_chart(hbar(top[["pair", "count"]], "count", "pair", color="#EF4444"))

        section("Confusion heatmap", "Focused on the intents involved in the pairs above.")
        involved = sorted(set(top["actual"]) | set(top["predicted"]))
        sub = cm.loc[[i for i in involved if i in cm.index], [i for i in involved if i in cm.columns]]
        heat = sub.stack().reset_index()
        heat.columns = ["actual", "predicted", "count"]
        heat["actual"] = heat["actual"].map(pretty)
        heat["predicted"] = heat["predicted"].map(pretty)
        heatmap = (
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted", axis=alt.Axis(labelAngle=-45, labelLimit=180)),
                y=alt.Y("actual:N", title="Actual", axis=alt.Axis(labelLimit=180)),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="purples"), legend=None),
                tooltip=["actual", "predicted", "count"],
            )
            .properties(height=max(320, 22 * len(involved)))
        )
        show_chart(heatmap)

# ------------------------------------------------------------------ Errors
with tab_err:
    section("Prediction error explorer", "Test messages where the predicted intent was wrong.")
    if errors.empty:
        st.success("No incorrect predictions were found.")
    else:
        c1, c2, c3 = st.columns([2, 2, 3])
        true_opts = ["All intents"] + sorted(errors["label"].unique())
        pred_opts = ["All intents"] + sorted(errors["predicted_label"].unique())
        sel_true = c1.selectbox("Correct intent", true_opts, format_func=lambda v: v if v == "All intents" else pretty(v))
        sel_pred = c2.selectbox("Predicted intent", pred_opts, format_func=lambda v: v if v == "All intents" else pretty(v))
        keyword = c3.text_input("Search message text")

        view = errors
        if sel_true != "All intents":
            view = view[view["label"] == sel_true]
        if sel_pred != "All intents":
            view = view[view["predicted_label"] == sel_pred]
        if keyword.strip():
            view = view[view["text"].str.contains(keyword.strip(), case=False, regex=False, na=False)]

        shown = view[["text", "label", "predicted_label"]].rename(
            columns={"text": "Customer message", "label": "Correct intent", "predicted_label": "Predicted intent"}
        )
        shown["Correct intent"] = shown["Correct intent"].map(pretty)
        shown["Predicted intent"] = shown["Predicted intent"].map(pretty)

        st.caption(f"Showing up to 200 of {len(shown):,} matching incorrect predictions.")
        table(shown.head(200))
        st.download_button(
            "Download errors (CSV)", shown.to_csv(index=False).encode("utf-8"),
            file_name="prediction_errors.csv", mime="text/csv", icon=":material/download:",
        )

st.info(
    "How to read this: accuracy measures overall correctness, macro-F1 weighs every intent "
    "equally, and per-intent metrics show where the classifier struggles. Benchmark results "
    "alone do not establish production readiness."
)