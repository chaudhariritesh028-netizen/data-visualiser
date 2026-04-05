import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import google.generativeai as genai

# ─────────────────────────────────────────────
#  PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AetherVisuals · AI 3D Data Architect",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar starts closed — cleaner first view
)

# ─────────────────────────────────────────────
#  API KEY — preloaded, falls back to sidebar
# ─────────────────────────────────────────────
PRELOADED_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyAu2uJLw3bzb1t96QdZjhNYcFMQzRwPQWQ")

# ─────────────────────────────────────────────
#  GLOBAL CSS — Glassmorphism Dark
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary:   #020510;
    --bg-glass:     rgba(5, 15, 40, 0.70);
    --border-glow:  rgba(0, 200, 255, 0.25);
    --accent-cyan:  #00d4ff;
    --accent-pink:  #ff2d78;
    --accent-lime:  #39ff14;
    --text-primary: #e0f4ff;
    --text-muted:   #5a7a9a;
    --font-display: 'Orbitron', monospace;
    --font-mono:    'Share Tech Mono', monospace;
    --font-body:    'Inter', sans-serif;
}

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,80,160,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(100,0,140,0.12) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,200,255,0.03) 39px, rgba(0,200,255,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,200,255,0.03) 39px, rgba(0,200,255,0.03) 40px);
    font-family: var(--font-body);
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; }

[data-testid="stSidebar"] {
    background: rgba(2, 8, 25, 0.97) !important;
    border-right: 1px solid var(--border-glow) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

[data-testid="collapsedControl"] {
    background: rgba(0,10,30,0.9) !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--accent-cyan) !important;
    top: 1.2rem !important;
}
[data-testid="collapsedControl"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.3) !important;
}

.aether-title {
    font-family: var(--font-display);
    font-size: clamp(1.8rem, 3vw, 2.8rem);
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--accent-cyan) 0%, #a78bfa 50%, var(--accent-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.aether-sub {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    opacity: 0.6;
}
.aether-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.4rem;
    border-bottom: 1px solid var(--border-glow);
    margin-bottom: 1.6rem;
}
.aether-header-left { display: flex; align-items: center; gap: 1.2rem; }
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    border: 1px solid rgba(57,255,20,0.5);
    color: var(--accent-lime);
    background: rgba(57,255,20,0.06);
    letter-spacing: 0.1em;
    white-space: nowrap;
}

.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    margin-bottom: 1rem;
    box-shadow: 0 0 30px rgba(0,180,255,0.05), inset 0 1px 0 rgba(255,255,255,0.05);
}

.section-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.6rem;
    opacity: 0.8;
}

.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.4rem; }
.metric-badge {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    border: 1px solid;
    white-space: nowrap;
}
.badge-ok   { border-color: var(--accent-lime); color: var(--accent-lime); background: rgba(57,255,20,0.08); }
.badge-warn { border-color: #ffb800;            color: #ffb800;            background: rgba(255,184,0,0.08); }
.badge-crit { border-color: var(--accent-pink); color: var(--accent-pink); background: rgba(255,45,120,0.08); }
.badge-info { border-color: var(--accent-cyan); color: var(--accent-cyan); background: rgba(0,212,255,0.08); }

.code-block {
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: #a8d8ff;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0.5rem 0;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    border: 1px solid var(--accent-lime);
    color: var(--accent-lime);
    background: rgba(57,255,20,0.07);
    letter-spacing: 0.1em;
}
.dot-pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-lime);
    box-shadow: 0 0 6px var(--accent-lime);
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: rgba(0,10,30,0.8) !important;
    border-color: rgba(0,200,255,0.3) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(0,100,180,0.5), rgba(80,0,140,0.5)) !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.35) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.3) !important;
    border-radius: 12px !important;
    background: rgba(0,10,30,0.5) !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-cyan) !important; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stAlert { border-radius: 8px !important; font-family: var(--font-mono) !important; font-size: 0.82rem !important; }

[data-baseweb="tab-list"] {
    background: rgba(0,10,30,0.6) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] { font-family: var(--font-display) !important; font-size: 0.65rem !important; letter-spacing: 0.12em !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(0,150,255,0.25), rgba(120,0,200,0.25)) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

.cyber-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 1.2rem 0;
    opacity: 0.4;
}
.sidebar-title {
    font-family: var(--font-display);
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.3); border-radius: 3px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "df_raw": None,
    "df_clean": None,
    "cleaning_history": [],
    "viz_code": [],
    "viz_figs": [],
    "model": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  AUTO-INIT MODEL from preloaded key
# ─────────────────────────────────────────────
if st.session_state.model is None and PRELOADED_API_KEY:
    try:
        genai.configure(api_key=PRELOADED_API_KEY)
        st.session_state.model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        st.session_state.model = None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()

def safe_exec_cleaning(code: str, df: pd.DataFrame):
    local_ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<cleaning_code>", "exec"), local_ns)  # noqa: S102
    return local_ns["df"]

def safe_exec_viz(code: str, df: pd.DataFrame):
    local_ns = {"df": df, "pd": pd, "np": np, "go": go, "px": px}
    exec(compile(code, "<viz_code>", "exec"), local_ns)  # noqa: S102
    return local_ns["create_figure"](df)

def ask_gemini(prompt: str, model) -> str:
    return model.generate_content(prompt).text

def health_report(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    return {
        "rows": df.shape[0], "cols": df.shape[1],
        "missing": missing,
        "missing_pct": round(missing / total_cells * 100, 1) if total_cells else 0,
        "duplicates": duplicates,
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

def badge_html(label, cls):
    return f'<span class="metric-badge {cls}">{label}</span>'

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⬡ System Configuration</p>', unsafe_allow_html=True)
    st.markdown('<div class="status-chip"><span class="dot-pulse"></span>AETHERVISUALS v1.0</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">🔑 API Key Override</p>', unsafe_allow_html=True)
    st.caption("A key is pre-loaded. Paste your own below to override.")
    override_key = st.text_input("Custom API Key", type="password", placeholder="AIza... (optional)", label_visibility="collapsed")
    if override_key and override_key != PRELOADED_API_KEY:
        try:
            genai.configure(api_key=override_key)
            st.session_state.model = genai.GenerativeModel("gemini-1.5-flash")
            st.success("✅ Custom key active")
        except Exception as e:
            st.error(f"⛔ {e}")

    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">📡 Pipeline Status</p>', unsafe_allow_html=True)
    for label, done in [
        ("AI Engine",    bool(st.session_state.model)),
        ("Data Loaded",  st.session_state.df_raw is not None),
        ("Data Cleaned", st.session_state.df_clean is not None),
        ("Viz Ready",    bool(st.session_state.viz_figs)),
    ]:
        st.markdown(f"{'🟢' if done else '⬜'} `{label}`")

    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:var(--font-mono);font-size:0.65rem;color:#3a5a7a;">Streamlit · Pandas · Gemini · Plotly</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
ai_badge = (
    '<span class="ai-badge"><span class="dot-pulse"></span>AI ONLINE</span>'
    if st.session_state.model else
    '<span class="ai-badge" style="border-color:rgba(255,45,120,0.5);color:#ff2d78;background:rgba(255,45,120,0.06);">⚠ AI OFFLINE</span>'
)
st.markdown(f"""
<div class="aether-header-bar">
    <div class="aether-header-left">
        <span style="font-size:2.4rem;">🌌</span>
        <div>
            <h1 class="aether-title">AetherVisuals</h1>
            <p class="aether-sub">AI-Powered · 3D Data Architect</p>
        </div>
    </div>
    {ai_badge}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SECTION 1 — UPLOAD & HEALTH REPORT
# ─────────────────────────────────────────────
col_upload, col_health = st.columns([1, 1.4], gap="medium")

with col_upload:
    st.markdown('<p class="section-label">◈ Data Ingestion Portal</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.session_state.df_raw = df_in.copy()
            if st.session_state.df_clean is None:
                st.session_state.df_clean = df_in.copy()
            st.success(f"✅ `{uploaded.name}` — {df_in.shape[0]:,} rows × {df_in.shape[1]} cols")
        except Exception as e:
            st.error(f"Parse error: {e}")

with col_health:
    st.markdown('<p class="section-label">◈ Data Health Report</p>', unsafe_allow_html=True)
    if st.session_state.df_raw is not None:
        rpt = health_report(st.session_state.df_clean)
        miss_cls = "badge-ok" if rpt["missing"] == 0 else ("badge-warn" if rpt["missing_pct"] < 10 else "badge-crit")
        dup_cls  = "badge-ok" if rpt["duplicates"] == 0 else "badge-warn"
        st.markdown(
            '<div class="metric-row">'
            + badge_html(f"📐 {rpt['rows']:,} rows", "badge-info")
            + badge_html(f"🗂 {rpt['cols']} cols", "badge-info")
            + badge_html(f"❓ {rpt['missing']} missing ({rpt['missing_pct']}%)", miss_cls)
            + badge_html(f"♻️ {rpt['duplicates']} duplicates", dup_cls)
            + "</div>", unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        miss_col = st.session_state.df_clean.isnull().sum()
        miss_col = miss_col[miss_col > 0]
        if not miss_col.empty:
            st.markdown("**Missing values per column:**")
            miss_df = miss_col.reset_index()
            miss_df.columns = ["Column", "Missing"]
            miss_df["% Missing"] = (miss_df["Missing"] / rpt["rows"] * 100).round(1)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<span class="metric-badge badge-ok">✓ No missing values detected</span>', unsafe_allow_html=True)
        dtype_str = " · ".join([f"`{str(k).split('.')[-1]}` ×{v}" for k, v in rpt["dtypes"].items()])
        st.caption(f"Column types: {dtype_str}")
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2rem;">'
            '<span style="font-size:2rem;">📂</span><br>'
            '<span style="font-family:var(--font-mono);font-size:0.8rem;color:#3a5a7a;">Upload a CSV to begin analysis</span></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
#  SECTION 2 — DATA PREVIEW
# ─────────────────────────────────────────────
if st.session_state.df_clean is not None:
    with st.expander("🔬 Data Preview & Statistics", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📋 Raw View", "📊 Describe", "🔭 Cleaned Head"])
        with tab1:
            st.dataframe(st.session_state.df_clean.head(50), use_container_width=True)
        with tab2:
            st.dataframe(st.session_state.df_clean.describe(include="all").T, use_container_width=True)
        with tab3:
            st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)
    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SECTION 3 — THE JANITOR CONSOLE
# ─────────────────────────────────────────────
if st.session_state.df_clean is not None:
    st.markdown('<p class="section-label">◈ The Janitor Console — Natural Language Data Cleaning</p>', unsafe_allow_html=True)
    col_j, col_h = st.columns([1.3, 1], gap="medium")

    with col_j:
        cleaning_cmd = st.text_area(
            "Command", height=120, label_visibility="collapsed",
            placeholder=(
                "e.g.  'Remove outliers in the Price column using IQR'\n"
                "      'Fill missing Age values with the median'\n"
                "      'Drop duplicate rows'\n"
                "      'Convert the Date column to datetime'"
            ),
        )
        c1, c2 = st.columns(2)
        run_clean = c1.button("⚡ Execute Command", use_container_width=True)
        reset_btn = c2.button("🔄 Reset to Raw",    use_container_width=True)

        if reset_btn and st.session_state.df_raw is not None:
            st.session_state.df_clean = st.session_state.df_raw.copy()
            st.session_state.cleaning_history = []
            st.success("Dataset reset to original.")

        if run_clean and cleaning_cmd.strip():
            if not st.session_state.model:
                st.error("⛔ AI engine offline.")
            else:
                with st.spinner("🤖 Generating Pandas code..."):
                    prompt = f"""You are a senior data-cleaning engineer.
DataFrame info:
- Columns & dtypes: {st.session_state.df_clean.dtypes.to_dict()}
- Sample rows:
{st.session_state.df_clean.head(5).to_string()}

User command: "{cleaning_cmd}"

Generate ONLY a ```python ... ``` code block that:
1. Operates on variable `df` (already a Pandas DataFrame).
2. Reassigns df or modifies in-place so `df` holds the result.
3. Uses only pandas (pd) and numpy (np) — already imported.
4. Does NOT include import statements.
Return ONLY the code block."""
                    code = extract_code(ask_gemini(prompt, st.session_state.model))
                    st.markdown('<p class="section-label">Generated Code</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="code-block">{code}</div>', unsafe_allow_html=True)

                    success, last_error, exec_code, attempt = False, None, code, 0
                    while attempt < 2 and not success:
                        try:
                            st.session_state.df_clean = safe_exec_cleaning(exec_code, st.session_state.df_clean)
                            st.session_state.cleaning_history.append({"command": cleaning_cmd, "code": exec_code, "status": "✅ success"})
                            success = True
                        except Exception as e:
                            last_error = str(e)
                            attempt += 1
                            if attempt < 2:
                                st.warning(f"⚠️ Error: {last_error}  \n🔁 Self-healing...")
                                exec_code = extract_code(ask_gemini(
                                    f"Fix this code:\n```python\n{exec_code}\n```\nError: {last_error}\nReturn ONLY the corrected ```python ... ``` block.",
                                    st.session_state.model
                                ))

                    if success:
                        st.success(f"✅ Done! DataFrame is now {st.session_state.df_clean.shape[0]:,} × {st.session_state.df_clean.shape[1]}")
                        st.rerun()
                    else:
                        st.error(f"❌ Self-healing failed.\n{last_error}")
                        st.session_state.cleaning_history.append({"command": cleaning_cmd, "code": exec_code, "status": f"❌ {last_error}"})

    with col_h:
        st.markdown('<p class="section-label">⟳ Cleaning History</p>', unsafe_allow_html=True)
        if st.session_state.cleaning_history:
            for i, entry in enumerate(reversed(st.session_state.cleaning_history[-8:])):
                st.markdown(
                    f'<div class="glass-card" style="padding:0.7rem 1rem;margin-bottom:0.5rem;">'
                    f'<span style="font-family:var(--font-mono);font-size:0.7rem;color:#5a7a9a;">#{len(st.session_state.cleaning_history)-i}</span> '
                    f'{entry["status"]}<br><span style="font-size:0.78rem;">{entry["command"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No cleaning operations yet.")

    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=st.session_state.df_clean.to_csv(index=False).encode(),
        file_name="aether_cleaned_data.csv",
        mime="text/csv",
    )
    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SECTION 4 — AI VISUALIZER ENGINE
# ─────────────────────────────────────────────
if st.session_state.df_clean is not None:
    st.markdown('<p class="section-label">◈ AI Visualizer Engine — 3D Holographic Renderer</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2], gap="medium")

    with col_l:
        st.markdown(
            '<div class="glass-card"><span style="font-family:var(--font-mono);font-size:0.78rem;color:#5a7a9a;">'
            "AetherVisuals will analyse your dataset and ask Gemini to propose the 3 most impactful "
            "3D visualisations, each rendered as a full interactive holographic chart.</span></div>",
            unsafe_allow_html=True,
        )
        generate_viz = st.button("🚀 Generate 3D Visualizations", use_container_width=True)
        if st.session_state.viz_figs:
            st.markdown(
                f'<div class="glass-card"><span class="metric-badge badge-ok">✓ {len(st.session_state.viz_figs)} Viz Ready</span></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        if generate_viz:
            if not st.session_state.model:
                st.error("⛔ AI engine offline.")
            else:
                df = st.session_state.df_clean
                with st.spinner("🌌 Consulting Gemini for 3D visualization blueprints..."):
                    master_prompt = f"""You are a world-class data visualization expert and Python engineer.

Dataset info:
- Columns & dtypes: {df.dtypes.to_dict()}
- Head:
{df.head(8).to_string()}
- Statistics:
{df.describe(include="all").to_string()}

Propose exactly 3 highly impactful 3D visualisations for this data.
For each, output a self-contained Python code block that:
1. Defines `def create_figure(df):` returning a plotly.graph_objects.Figure.
2. Uses ONLY `go`, `pd`, `np` (already imported).
3. Always uses 3D axes (scene with xaxis, yaxis, zaxis). No 2D charts.
4. Uses template="plotly_dark".
5. Uses colorscale="Viridis" or colorscale="Electric".
6. Sets paper_bgcolor="rgba(0,0,0,0)" and plot_bgcolor="rgba(0,0,0,0)".
7. Adds lighting=dict(ambient=0.4, diffuse=0.6, specular=0.8, roughness=0.5, fresnel=0.8) where supported.
8. Sets a descriptive title.
9. All preprocessing inside the function. Handles errors with try/except.

When generating Plotly code, always use 3D axes (x, y, z). Focus on depth and lighting. Return ONLY the python code block, no conversational text.

Output exactly 3 ```python ... ``` blocks with no other text."""

                    raw = ask_gemini(master_prompt, st.session_state.model)

                all_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", raw)
                viz_blocks = [b.strip() for b in all_blocks if "create_figure" in b] or [raw.strip()]

                st.session_state.viz_code  = viz_blocks
                st.session_state.viz_figs  = []

                for i, block in enumerate(viz_blocks[:3]):
                    fig, exec_block, attempt = None, block, 0
                    while attempt < 2 and fig is None:
                        try:
                            fig = safe_exec_viz(exec_block, df)
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Share Tech Mono, monospace", color="#c0e0ff"),
                                margin=dict(l=20, r=20, t=50, b=20),
                                scene=dict(
                                    bgcolor="rgba(0,0,0,0)",
                                    xaxis=dict(backgroundcolor="rgba(0,10,30,0.6)",  gridcolor="rgba(0,180,255,0.15)",  showbackground=True, zerolinecolor="rgba(0,200,255,0.2)",  tickfont=dict(size=9)),
                                    yaxis=dict(backgroundcolor="rgba(5,0,20,0.6)",   gridcolor="rgba(150,0,255,0.15)",  showbackground=True, zerolinecolor="rgba(150,0,255,0.2)", tickfont=dict(size=9)),
                                    zaxis=dict(backgroundcolor="rgba(0,5,20,0.6)",   gridcolor="rgba(0,100,200,0.15)",  showbackground=True, zerolinecolor="rgba(0,150,255,0.2)", tickfont=dict(size=9)),
                                ),
                            )
                            st.session_state.viz_figs.append(fig)
                        except Exception as e:
                            attempt += 1
                            if attempt < 2 and st.session_state.model:
                                exec_block = extract_code(ask_gemini(
                                    f"Fix this Plotly code:\n```python\n{exec_block}\n```\nError: {e}\n"
                                    "Function must be `def create_figure(df):` returning go.Figure.\n"
                                    "Return ONLY the corrected ```python ... ``` block.",
                                    st.session_state.model
                                ))
                            else:
                                st.warning(f"⚠️ Visualization {i+1} could not be rendered: {e}")
                st.rerun()

    if st.session_state.viz_figs:
        tabs = st.tabs([f"🌐 Viz {i+1}" for i in range(len(st.session_state.viz_figs))])
        for i, (tab, fig) in enumerate(zip(tabs, st.session_state.viz_figs)):
            with tab:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                with st.expander(f"🧬 Source Code — Visualization {i+1}"):
                    if i < len(st.session_state.viz_code):
                        st.markdown(f'<div class="code-block">{st.session_state.viz_code[i]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🌌</div>
            <div style="font-family:var(--font-display);font-size:1.1rem;letter-spacing:0.2em;
                        background:linear-gradient(135deg,#00d4ff,#a78bfa,#ff2d78);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                AWAITING DATA STREAM
            </div>
            <div style="font-family:var(--font-mono);font-size:0.78rem;color:#3a5a7a;margin-top:0.8rem;">
                Upload a CSV file to initialise the AetherVisuals pipeline
            </div>
        </div>
    """, unsafe_allow_html=True)
