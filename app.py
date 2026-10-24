import streamlit as st

from ui.data import pretty, read_json
from ui.theme import inject_css, status_card

st.set_page_config(
    page_title="Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

navigation = st.navigation(
    {
        "Workspace": [
            st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page("views/ticket_triage.py", title="Ticket Triage", icon=":material/confirmation_number:"),
            st.Page("views/department_workload.py", title="Department Workload", icon=":material/groups:"),
        ],
        "Insights": [
            st.Page("views/support_analytics.py", title="Support Analytics", icon=":material/bar_chart:"),
            st.Page("views/model_evaluation.py", title="Model Evaluation", icon=":material/science:"),
        ],
    }
)

# Sidebar system status.
with st.sidebar:
    try:
        settings = read_json("routing_settings.json")
        summary = read_json("evaluation_summary.json")
        queue = st.session_state.get("review_queue", [])
        pending = sum(1 for t in queue if t["status"] == "Pending Review")
        status_card(
            "System status",
            [
                ("Model", pretty(settings.get("selected_model", "n/a"))),
                ("Threshold", f"{settings['confidence_threshold']:.0%}"),
                ("Test accuracy", f"{summary['accuracy']:.1%}"),
                ("Pending reviews", pending),
            ],
        )
    except Exception:
        pass
    st.caption("BANKING77 benchmark · prototype")

navigation.run()