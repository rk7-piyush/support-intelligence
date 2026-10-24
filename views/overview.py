import streamlit as st

from ui.data import ARTIFACTS, read_json, require
from ui.theme import kpi_row, page_header, section

require(ARTIFACTS / "evaluation_summary.json", ARTIFACTS / "routing_settings.json")

summary = read_json("evaluation_summary.json")
settings = read_json("routing_settings.json")

page_header(
    "Support Intelligence System",
    "Classify customer issues, route them to the right department, and send "
    "uncertain predictions to human review.",
    eyebrow="Overview",
)

queue = st.session_state.get("review_queue", [])
pending = sum(1 for t in queue if t["status"] == "Pending Review")

kpi_row(
    [
        {"label": "Test accuracy", "value": f"{summary['accuracy']:.1%}",
         "sub": f"{summary['test_examples']:,} held-out examples", "tone": "indigo"},
        {"label": "Macro-F1", "value": f"{summary['macro_f1']:.3f}",
         "sub": f"{summary['number_of_intents']} support intents", "tone": "green"},
        {"label": "Auto-route threshold", "value": f"{settings['confidence_threshold']:.0%}",
         "sub": f"{settings['validation_coverage']:.0%} of tickets auto-routed (validation)", "tone": "amber"},
        {"label": "Pending reviews", "value": pending,
         "sub": "In this session's queue", "tone": "slate"},
    ]
)

section("Workspaces", "Choose where you want to work.")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### 🎫 Ticket Triage")
        st.write(
            "Classify a message, see the predicted intent and confidence, "
            "triage in batches, and work the manual-review queue."
        )
        st.page_link("views/ticket_triage.py", label="Open Ticket Triage", icon=":material/arrow_forward:", width="stretch")

with col2:
    with st.container(border=True):
        st.markdown("#### 📊 Support Analytics")
        st.write(
            "Explore intent and department distribution, browse example "
            "messages, and search the dataset."
        )
        st.page_link("views/support_analytics.py", label="Open Support Analytics", icon=":material/arrow_forward:", width="stretch")

with col3:
    with st.container(border=True):
        st.markdown("#### 🧪 Model Evaluation")
        st.write(
            "Review accuracy and F1, tune the confidence threshold, and "
            "investigate confusing intents and errors."
        )
        st.page_link("views/model_evaluation.py", label="Open Model Evaluation", icon=":material/arrow_forward:", width="stretch")

section("How it works")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("**01 · Classify**")
    st.write("A calibrated TF-IDF + Linear SVM model predicts the most likely support intent.")
with c2:
    st.markdown("**02 · Decide**")
    st.write("Confidence is compared with the threshold: auto-route, or send to manual review.")
with c3:
    st.markdown("**03 · Evaluate**")
    st.write("Saved benchmark results reveal overall performance and weak categories.")

st.info(
    "This project uses the BANKING77 benchmark, not live bank data. The review "
    "queue lives in the current browser session and is not persistent storage. "
    "Model confidence is an estimate, not a guarantee of correctness."
)