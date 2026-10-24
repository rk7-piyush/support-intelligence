"""Shared look & feel: CSS, layout helpers and chart styling."""
import html

import altair as alt
import streamlit as st

PRIMARY = "#4F46E5"
GREEN = "#059669"
AMBER = "#D97706"
RED = "#DC2626"
SLATE = "#64748B"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }



/* Sections / bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 0.6rem 1rem; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }

/* Page header */
.si-header { margin-bottom: 1.4rem; }
.si-eyebrow { color: #4F46E5; font-size: .78rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem; }
.si-title { color: #111827; font-size: 2rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.2; }
.si-sub { color: #6B7280; font-size: 1.02rem; line-height: 1.6; margin: .5rem 0 0 0; max-width: 820px; }

/* KPI cards */
.si-kpi { background: #FFFFFF; border: 1px solid #E5E7EB; border-top: 3px solid #4F46E5;
  border-radius: 14px; padding: 1rem 1.15rem; box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%; }
.si-kpi.si-green { border-top-color: #059669; }
.si-kpi.si-amber { border-top-color: #D97706; }
.si-kpi.si-red { border-top-color: #DC2626; }
.si-kpi.si-slate { border-top-color: #64748B; }
.si-kpi-label { color: #6B7280; font-size: .78rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em; }
.si-kpi-value { color: #111827; font-size: 1.75rem; font-weight: 700; margin-top: .25rem; line-height: 1.2; }
.si-kpi-sub { color: #6B7280; font-size: .82rem; margin-top: .2rem; }

/* Section titles */
.si-section { margin: 1.4rem 0 .6rem 0; }
.si-section-title { color: #111827; font-size: 1.15rem; font-weight: 700; margin: 0; }
.si-section-cap { color: #6B7280; font-size: .9rem; margin-top: .15rem; }

/* Decision banner */
.si-banner { border-radius: 14px; padding: 1rem 1.2rem; border: 1px solid; margin: .4rem 0 1rem 0; }
.si-banner.ok { background: #ECFDF5; border-color: #A7F3D0; }
.si-banner.warn { background: #FFFBEB; border-color: #FDE68A; }
.si-banner-title { font-size: 1.1rem; font-weight: 700; }
.si-banner.ok .si-banner-title { color: #065F46; }
.si-banner.warn .si-banner-title { color: #92400E; }
.si-banner-body { color: #374151; font-size: .93rem; margin-top: .2rem; }

/* Badges */
.si-badge { display: inline-block; padding: .15rem .6rem; border-radius: 999px;
  font-size: .75rem; font-weight: 600; margin-right: .35rem; }
.si-badge.ok { background: #D1FAE5; color: #065F46; }
.si-badge.warn { background: #FEF3C7; color: #92400E; }
.si-badge.info { background: #E0E7FF; color: #3730A3; }
.si-badge.muted { background: #F1F5F9; color: #475569; }

/* Confidence meter */
.si-meter-track { position: relative; height: 12px; background: #E5E7EB; border-radius: 999px; margin-top: .6rem; }
.si-meter-fill { height: 100%; border-radius: 999px; }
.si-meter-fill.ok { background: #059669; }
.si-meter-fill.warn { background: #D97706; }
.si-meter-thr { position: absolute; top: -5px; width: 3px; height: 22px; background: #111827; border-radius: 2px; }
.si-meter-legend { display: flex; justify-content: space-between; color: #6B7280; font-size: .78rem; margin-top: .4rem; }

/* Quote + status card */
.si-quote { background: #F8FAFC; border-left: 3px solid #4F46E5; border-radius: 8px;
  padding: .7rem .9rem; color: #1F2937; font-size: .95rem; margin: .3rem 0 .6rem 0; }
.si-card { background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 12px; padding: .8rem .9rem; margin-top: 1rem; }
.si-card-title { color: #6B7280; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
.si-card-row { display: flex; justify-content: space-between; color: #374151; font-size: .85rem; margin-top: .35rem; }
.si-card-row b { color: #111827; }


/* ---- Theme-safe overrides (work in light AND dark mode) ---- */
.si-eyebrow { color: #818CF8; }
.si-title, .si-section-title { color: inherit !important; }
.si-sub, .si-section-cap { color: inherit !important; opacity: .72; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(128,128,128,.3); }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def page_header(title, subtitle, eyebrow="Support Intelligence"):
    st.markdown(
        f'<div class="si-header"><div class="si-eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="si-title">{esc(title)}</h1>'
        f'<p class="si-sub">{esc(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, caption=None):
    cap = f'<div class="si-section-cap">{esc(caption)}</div>' if caption else ""
    st.markdown(
        f'<div class="si-section"><div class="si-section-title">{esc(title)}</div>{cap}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of dicts with label, value, optional sub and tone
    (indigo | green | amber | red | slate)."""
    cols = st.columns(len(items), gap="medium")
    for col, item in zip(cols, items):
        tone = item.get("tone", "indigo")
        sub = f'<div class="si-kpi-sub">{esc(item["sub"])}</div>' if item.get("sub") else ""
        col.markdown(
            f'<div class="si-kpi si-{tone}">'
            f'<div class="si-kpi-label">{esc(item["label"])}</div>'
            f'<div class="si-kpi-value">{esc(item["value"])}</div>{sub}</div>',
            unsafe_allow_html=True,
        )


def badge(text, tone="info") -> str:
    return f'<span class="si-badge {tone}">{esc(text)}</span>'


def quote(text) -> str:
    return f'<div class="si-quote">{esc(text)}</div>'


def decision_banner(auto: bool, title: str, body: str):
    cls = "ok" if auto else "warn"
    st.markdown(
        f'<div class="si-banner {cls}"><div class="si-banner-title">{esc(title)}</div>'
        f'<div class="si-banner-body">{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def confidence_meter(conf: float, thr: float):
    cls = "ok" if conf >= thr else "warn"
    st.markdown(
        f'<div class="si-meter-track"><div class="si-meter-fill {cls}" style="width:{conf*100:.1f}%"></div>'
        f'<div class="si-meter-thr" style="left:{thr*100:.1f}%"></div></div>'
        f'<div class="si-meter-legend"><span>0%</span><span>Threshold {thr:.0%}</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )


def status_card(title, rows):
    body = "".join(f'<div class="si-card-row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in rows)
    st.markdown(
        f'<div class="si-card"><div class="si-card-title">{esc(title)}</div>{body}</div>',
        unsafe_allow_html=True,
    )


# ---------- Charts & tables (one place to adjust if Streamlit APIs change) ----------

def _style(chart):
    muted = "#94A3B8"  # readable on both light and dark backgrounds
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=muted, titleColor=muted, gridColor="rgba(148,163,184,0.25)",
            domainColor="rgba(148,163,184,0.4)", tickColor="rgba(148,163,184,0.4)",
            labelFontSize=12,
        )
        .configure_legend(labelColor=muted, orient="top", title=None)
    )


def show_chart(chart):
    st.altair_chart(_style(chart), width="stretch", theme=None)


def hbar(df, value, category, color=PRIMARY, domain=None, fmt=",.0f", height=None):
    scale = alt.Scale(domain=list(domain)) if domain else alt.Undefined
    base = alt.Chart(df).encode(
        x=alt.X(f"{value}:Q", title=None, scale=scale),
        y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4).encode(
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=fmt)]
    )
    labels = base.mark_text(align="left", dx=4, color="#94A3B8", fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return (bars + labels).properties(height=height or max(200, 28 * len(df)))


def table(df, **kwargs):
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)