"""Cached data loaders shared by every page."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.routing import get_department, load_routing_config
from src.letter_triage import extra_departments

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
CONFIG = ROOT / "config" / "intent_routing.json"


def pretty(value) -> str:
    return str(value).replace("_", " ").strip().title()


def require(*paths):
    """Stop the page with a friendly message if required files are missing."""
    missing = [Path(p).name for p in paths if not Path(p).exists()]
    if missing:
        st.error("Required files are missing: " + ", ".join(missing))
        st.info(
            "Make sure the `data/` and `artifacts/` folders are committed to "
            "your repository (check `.gitignore`)."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    import json

    with open(ARTIFACTS / name, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def read_artifact_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name, **kwargs)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(ARTIFACTS / "best_model.joblib")


@st.cache_data(show_spinner=False)
def routing_config() -> dict:
    return load_routing_config()


@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv").dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str)
    cfg = routing_config()
    df["department"] = df["label"].map(lambda label: get_department(label, cfg))
    df["words"] = df["text"].astype(str).str.split().str.len()
    return df


def department_list() -> list:
    cfg = routing_config()
    names = {rule["department"] for rule in cfg["rules"]}
    names.add(cfg["default_department"])
    names.update(extra_departments())
    return sorted(names)