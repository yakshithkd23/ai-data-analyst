"""
Ledger — AI Data Analyst
Upload CSVs, ask questions in plain English, get charts, SQL/Pandas code,
anomaly detection and data-quality checks. Powered by Groq + LangChain.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ledger · AI Data Analyst", page_icon="📈", layout="wide")

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

MODELS = ["llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-20b"]

SUGGESTIONS = [
    ("Top performers", "Which region generated the highest revenue?"),
    ("Trends", "Show monthly sales trends as a line chart."),
    ("Weak spots", "Which products are underperforming?"),
    ("Customers", "What are the top five customers by revenue?"),
    ("SQL", "Generate SQL for revenue by region."),
    ("Anomalies", "Detect anomalies in the dataset and explain why they were flagged."),
]

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
INK, BLUE, SIGNAL, PAPER, LINE = "#14213D", "#3256D6", "#E8A317", "#F2F5F9", "#DDE3EC"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
code, pre {{ font-family: 'JetBrains Mono', monospace !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E8ECF5; }}
[data-testid="stSidebar"] input {{ color: {INK} !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,.06); border: 1px dashed rgba(255,255,255,.3); border-radius: 12px;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {{ color: #E8ECF5 !important; }}
[data-testid="stSidebar"] button {{ color: {INK} !important; }}
.brand {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; }}
.brand-sub {{ font-size: .85rem; opacity: .7; margin-bottom: 1.2rem; }}

/* Hero */
.hero {{
    background: {INK}; color: #fff; border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.2rem;
    border-left: 8px solid {SIGNAL};
}}
.hero h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -.03em; margin: 0 0 .4rem 0; color: #fff; }}
.hero p {{ margin: 0; opacity: .8; font-size: 1rem; max-width: 640px; line-height: 1.55; }}

/* Stat tiles */
[data-testid="stMetric"] {{
    background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: .9rem 1.1rem;
}}
[data-testid="stMetricLabel"] p {{ font-size: .82rem; color: #5B6785; }}
[data-testid="stMetricValue"] {{ font-weight: 800; letter-spacing: -.02em; }}

/* Nav */
div[role="radiogroup"] {{ gap: .4rem; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{
    background: #fff; border: 1px solid {LINE}; border-radius: 999px; padding: .35rem 1rem; cursor: pointer;
}}
div[role="radiogroup"] > label:has(input:checked) {{ background: {BLUE}; border-color: {BLUE}; }}
div[role="radiogroup"] > label:has(input:checked) * {{ color: #fff !important; }}
div[role="radiogroup"] > label > div:first-child {{ display: none; }}

/* Chat */
[data-testid="stChatMessage"] {{
    background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.2rem;
}}
[data-testid="stChatInput"] textarea {{ font-family: 'Plus Jakarta Sans', sans-serif; }}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    border-radius: 10px; font-weight: 600; border: 1px solid {LINE}; background: #fff; color: {INK};
}}
.stButton > button:hover {{ border-color: {BLUE}; color: {BLUE}; }}
.stButton > button:focus-visible {{ outline: 3px solid {BLUE}; outline-offset: 2px; }}

.empty {{
    background: #fff; border: 1px dashed {LINE}; border-radius: 16px; padding: 2.2rem; text-align: center;
}}
.empty h3 {{ margin: 0 0 .4rem 0; }}
.flag {{ border-left: 4px solid {SIGNAL}; background: #FFF7E6; padding: .7rem 1rem; border-radius: 8px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("datasets", {})
st.session_state.setdefault("pending", None)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(file) -> pd.DataFrame:
    """Read and validate an uploaded CSV."""
    df = pd.read_csv(file)
    if df.shape[1] == 0 or df.empty:
        raise ValueError("The file has no data rows.")
    for col in df.columns:
        if df[col].dtype == object and "date" in str(col).lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def sample_sales() -> pd.DataFrame:
    """Synthetic sales data with a few planted outliers and gaps."""
    rng = np.random.default_rng(42)
    n = 600
    products = {"Aurora Laptop": 950, "Pulse Phone": 600, "Nimbus Tablet": 420,
                "Echo Speaker": 120, "Halo Watch": 250, "Spark Charger": 30}
    names = rng.choice(list(products), n, p=[.12, .2, .15, .2, .13, .2])
    units = rng.integers(1, 8, n)
    price = np.array([products[p] for p in names]) * rng.uniform(.92, 1.08, n)
    df = pd.DataFrame({
        "order_date": rng.choice(pd.date_range("2025-01-01", "2025-12-31"), n),
        "region": rng.choice(["North", "South", "East", "West"], n, p=[.32, .22, .26, .2]),
        "product": names,
        "customer": rng.choice([f"Customer {i:02d}" for i in range(1, 41)], n),
        "units": units,
        "unit_price": price.round(2),
    })
    df["revenue"] = (df.units * df.unit_price).round(2)
    df["discount_pct"] = rng.choice([0, 5, 10, 15, np.nan], n, p=[.4, .2, .2, .1, .1])
    spikes = rng.choice(n, 6, replace=False)
    df.loc[spikes, "revenue"] *= 9
    return df.sort_values("order_date").reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> dict:
    cells = max(df.size, 1)
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    score = max(0, round(100 - (missing / cells) * 200 - (dups / max(len(df), 1)) * 200))
    return {"missing": missing, "duplicates": dups, "score": score}


def find_anomalies(df: pd.DataFrame, col: str, k: float = 1.5):
    """IQR rule. Returns flagged rows with a plain-language reason."""
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    flagged = df[(df[col] < lo) | (df[col] > hi)].copy()
    flagged["why_flagged"] = flagged[col].map(
        lambda v: f"{col} = {v:,.2f} is {'above' if v > hi else 'below'} the normal range "
                  f"({lo:,.2f} to {hi:,.2f})"
    )
    return flagged, lo, hi


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
def run_agent(question: str, api_key: str, model: str):
    """Ask the pandas agent, passing recent chat turns so follow-ups work."""
    from langchain_groq import ChatGroq
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain.agents.agent_types import AgentType

    dfs = list(st.session_state.datasets.values())
    llm = ChatGroq(temperature=0, model_name=model, groq_api_key=api_key, max_tokens=1200)
    
    # Switch to ZERO_SHOT_REACT_DESCRIPTION for better compatibility across various LLMs
    agent = create_pandas_dataframe_agent(
        llm, 
        dfs if len(dfs) > 1 else dfs[0], 
        verbose=False, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True, 
        return_intermediate_steps=True, 
        handle_parsing_errors=True,
    )
    
    chart_path = CHART_DIR / f"{uuid.uuid4().hex}.png"
    history = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in st.session_state.messages[-7:-1])
    
    prompt = f"""You are an expert data analyst. Be concise and business-focused.
Conversation so far: {history or '(none)'}
Current question: "{question}"
Rules:
- Start with the direct answer, then a short "Why" explaining your reasoning.
- If a chart is requested, build it with matplotlib, call plt.savefig(r'{chart_path}', bbox_inches='tight', dpi=150), then describe it.
- For anomalies, use IQR or Z-score, state the threshold, and list the flagged rows.
- If SQL is requested, give the SQL query in a ```sql block next to the pandas result.
"""
    try:
        result = agent.invoke({"input": prompt})
    except Exception as e:
        return f"Error executing agent: {e}", None, []

    code = []
    for step in result.get("intermediate_steps", []):
        tool_input = getattr(step[0], "tool_input", "")
        code.append(tool_input.get("query", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input))
        
    return result["output"], (str(chart_path) if chart_path.exists() else None), code

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Ledger</div><div class="brand-sub">Ask your data anything</div>',
                unsafe_allow_html=True)

    st.markdown("**1 · Connect**")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""),
                            placeholder="gsk_…")
    model = st.selectbox("Model", MODELS)

    st.markdown("**2 · Add data**")
    uploads = st.file_uploader("CSV files", type="csv", accept_multiple_files=True,
                               label_visibility="collapsed")
    if uploads:
        loaded = {}
        for f in uploads:
            try:
                loaded[f.name] = load_csv(f)
            except Exception as e:
                st.error(f"{f.name} could not be read: {e}")
        if loaded:
            st.session_state.datasets = loaded

    if st.button("Try the sample sales data", use_container_width=True):
        st.session_state.datasets = {"sample_sales.csv": sample_sales()}
        st.session_state.messages = []
        st.rerun()

    if st.session_state.datasets:
        st.markdown("**Loaded**")
        for name, d in st.session_state.datasets.items():
            st.caption(f"{name} · {len(d):,} rows × {d.shape[1]} cols")

    if st.session_state.messages:
        st.markdown("**3 · Share**")
        report = "# Ledger analysis report\n\n" + "\n\n".join(
            f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
        st.download_button("Download report (.md)", report,
                           file_name=f"analysis_{datetime.now():%Y%m%d_%H%M}.md", use_container_width=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="hero"><h1>Your data, in plain English</h1>
    <p>Upload a CSV and ask a question. Ledger answers, draws the chart, writes the SQL,
    flags unusual rows and shows how it got there.</p></div>""",
    unsafe_allow_html=True,
)

datasets = st.session_state.datasets

if not datasets:
    st.markdown(
        """<div class="empty"><h3>Start with a CSV</h3>
        Drop one or more files in the sidebar, or load the sample sales data to see how it works.</div>""",
        unsafe_allow_html=True,
    )
    st.stop()

active_name = st.selectbox("Dataset", list(datasets), label_visibility="collapsed") if len(datasets) > 1 \
    else next(iter(datasets))
df = datasets[active_name]
q = quality_report(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Files", len(datasets))
c2.metric("Rows", f"{len(df):,}")
c3.metric("Columns", df.shape[1])
c4.metric("Data quality", f"{q['score']}/100")

page = st.radio("View", ["Ask", "Explore", "Charts", "Anomalies", "Data quality"],
                horizontal=True, label_visibility="collapsed")
numeric_cols = df.select_dtypes("number").columns.tolist()
all_cols = df.columns.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# Ask
# ──────────────────────────────────────────────────────────────────────────────
if page == "Ask":
    if not st.session_state.messages:
        st.markdown("##### Try one of these")
        cols = st.columns(3)
        for i, (label, text) in enumerate(SUGGESTIONS):
            if cols[i % 3].button(f"{label}: {text}", key=f"s{i}", use_container_width=True):
                st.session_state.pending = text
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "📈"):
            st.markdown(m["content"])
            if m.get("image") and os.path.exists(m["image"]):
                st.image(m["image"])
            if m.get("code"):
                with st.expander("How I got this (code)"):
                    for block in m["code"]:
                        st.code(block, language="python")

    prompt = st.chat_input("Ask about your data, e.g. “Which region has the highest revenue?”") \
        or st.session_state.pop("pending", None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="📈"):
            if not api_key:
                msg = "Add your Groq API key in the sidebar to start asking questions."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                try:
                    with st.spinner("Reading your data and working out the answer…"):
                        output, image, code = run_agent(prompt, api_key, model)
                    st.markdown(output)
                    if image:
                        st.image(image)
                    if code:
                        with st.expander("How I got this (code)"):
                            for block in code:
                                st.code(block, language="python")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output, "image": image, "code": code})
                except Exception as e:
                    msg = (f"The analysis failed: {e}. Check your API key and model, "
                           "or try rephrasing the question.")
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

# ──────────────────────────────────────────────────────────────────────────────
# Explore
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Explore":
    st.dataframe(df, use_container_width=True, height=360)
    left, right = st.columns(2)
    with left:
        st.markdown("##### Column types")
        st.dataframe(pd.DataFrame({"column": df.columns, "type": df.dtypes.astype(str).values,
                                   "unique": df.nunique().values}), hide_index=True, use_container_width=True)
    with right:
        st.markdown("##### Summary statistics")
        st.dataframe(df.describe().T, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Charts (no API key needed)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Charts":
    a, b, c, d = st.columns(4)
    kind = a.selectbox("Chart", ["Bar", "Line", "Pie", "Scatter", "Histogram"])
    x = b.selectbox("X axis", all_cols)
    y = c.selectbox("Y axis", numeric_cols or all_cols)
    agg = d.selectbox("Aggregate", ["sum", "mean", "count"])

    try:
        if kind == "Scatter":
            fig = px.scatter(df, x=x, y=y)
        elif kind == "Histogram":
            fig = px.histogram(df, x=y)
        else:
            data = df.groupby(x, as_index=False)[y].agg(agg)
            if kind == "Bar":
                fig = px.bar(data, x=x, y=y)
            elif kind == "Line":
                fig = px.line(data.sort_values(x), x=x, y=y, markers=True)
            else:
                fig = px.pie(data, names=x, values=y)
        fig.update_traces(marker_color=BLUE) if kind in ("Bar", "Scatter", "Histogram") else None
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font_family="Plus Jakarta Sans",
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"This combination can't be charted: {e}. Pick a different column or aggregate.")

# ──────────────────────────────────────────────────────────────────────────────
# Anomalies (no API key needed)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Anomalies":
    if not numeric_cols:
        st.info("This dataset has no numeric columns to check.")
    else:
        a, b = st.columns([2, 1])
        col = a.selectbox("Column to check", numeric_cols,
                          index=numeric_cols.index("revenue") if "revenue" in numeric_cols else 0)
        k = b.slider("Sensitivity (lower = stricter)", 1.0, 3.0, 1.5, 0.5)
        flagged, lo, hi = find_anomalies(df, col, k)

        st.markdown(
            f'<div class="flag"><b>{len(flagged)} unusual rows</b> out of {len(df):,}. '
            f'Values outside {lo:,.2f} to {hi:,.2f} are flagged using the IQR rule.</div>',
            unsafe_allow_html=True)

        plot = df.assign(status=np.where(df.index.isin(flagged.index), "Flagged", "Normal"))
        fig = px.scatter(plot, x=plot.index, y=col, color="status",
                         color_discrete_map={"Normal": BLUE, "Flagged": SIGNAL},
                         labels={"x": "Row", col: col})
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font_family="Plus Jakarta Sans",
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        if len(flagged):
            st.dataframe(flagged, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data quality
# ──────────────────────────────────────────────────────────────────────────────
else:
    a, b, c = st.columns(3)
    a.metric("Quality score", f"{q['score']}/100")
    b.metric("Missing cells", f"{q['missing']:,}")
    c.metric("Duplicate rows", f"{q['duplicates']:,}")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.markdown("##### Missing values by column")
        fig = px.bar(missing.reset_index(), x="index", y=0, labels={"index": "Column", "0": "Missing"})
        fig.update_traces(marker_color=SIGNAL)
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font_family="Plus Jakarta Sans",
                          margin=dict(l=20, r=20, t=10, b=20), yaxis_title="Missing cells")
        st.plotly_chart(fig, use_container_width=True)