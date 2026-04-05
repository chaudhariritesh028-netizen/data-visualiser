import streamlit as st
import pandas as pd
import numpy as np
import re

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="AetherVisuals · AI 3D Data Architect",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lazy-import heavy packages so Streamlit Cloud can catch missing deps early ─
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    st.error("❌ Plotly is not installed. Check your requirements.txt.")
    st.stop()

try:
    import google.generativeai as genai
    GENAI_OK = True
except ImportError:
    GENAI_OK = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:          #020510;
    --glass:       rgba(5,15,40,0.75);
    --glow:        rgba(0,200,255,0.22);
    --cyan:        #00d4ff;
    --pink:        #ff2d78;
    --lime:        #39ff14;
    --purple:      #a78bfa;
    --text:        #e0f4ff;
    --muted:       #4a6a8a;
    --display:     'Orbitron', monospace;
    --mono:        'Share Tech Mono', monospace;
    --body:        'Inter', sans-serif;
}

/* ── Base ── */
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 10%,  rgba(0,80,160,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 90%,  rgba(100,0,140,0.14) 0%, transparent 60%),
        repeating-linear-gradient(0deg,   transparent, transparent 39px, rgba(0,200,255,0.025) 40px),
        repeating-linear-gradient(90deg,  transparent, transparent 39px, rgba(0,200,255,0.025) 40px);
    font-family: var(--body);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(1,5,18,0.98) !important;
    border-right: 1px solid var(--glow) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="collapsedControl"] {
    background: rgba(0,8,24,0.95) !important;
    border: 1px solid rgba(0,200,255,0.35) !important;
    border-radius: 0 8px 8px 0 !important;
    top: 1rem !important;
}
[data-testid="collapsedControl"]:hover {
    box-shadow: 0 0 14px rgba(0,212,255,0.4) !important;
}

/* ── Typography helpers ── */
.title {
    font-family: var(--display);
    font-size: clamp(1.6rem, 2.8vw, 2.6rem);
    font-weight: 900;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 55%, var(--pink) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.1;
}
.subtitle {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.section-label {
    font-family: var(--display);
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.85;
    margin: 0 0 0.5rem;
}
.sidebar-label {
    font-family: var(--display);
    font-size: 0.58rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.75;
    margin: 0 0 0.4rem;
}

/* ── Cards ── */
.glass {
    background: var(--glass);
    border: 1px solid var(--glow);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 0 28px rgba(0,180,255,0.04), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 0.9rem;
}
.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #80c8e8;
    margin-bottom: 0.8rem;
}
.warn-box {
    background: rgba(255,184,0,0.06);
    border: 1px solid rgba(255,184,0,0.25);
    border-left: 3px solid #ffb800;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #e8c870;
    margin-bottom: 0.8rem;
}
.success-box {
    background: rgba(57,255,20,0.05);
    border: 1px solid rgba(57,255,20,0.2);
    border-left: 3px solid var(--lime);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #90e870;
    margin-bottom: 0.8rem;
}

/* ── Badges ── */
.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
.badge {
    font-family: var(--mono); font-size: 0.74rem;
    padding: 0.3rem 0.7rem; border-radius: 6px; border: 1px solid; white-space: nowrap;
}
.b-ok   { border-color: var(--lime);   color: var(--lime);   background: rgba(57,255,20,0.07); }
.b-warn { border-color: #ffb800;       color: #ffb800;       background: rgba(255,184,0,0.07); }
.b-bad  { border-color: var(--pink);   color: var(--pink);   background: rgba(255,45,120,0.07); }
.b-info { border-color: var(--cyan);   color: var(--cyan);   background: rgba(0,212,255,0.07); }
.b-off  { border-color: var(--muted);  color: var(--muted);  background: rgba(74,106,138,0.07); }

/* ── Status pill ── */
.pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: var(--mono); font-size: 0.68rem;
    padding: 0.22rem 0.55rem; border-radius: 20px; letter-spacing: 0.08em;
}
.pill-on  { border: 1px solid rgba(57,255,20,0.5);  color: var(--lime);   background: rgba(57,255,20,0.06); }
.pill-off { border: 1px solid rgba(255,45,120,0.4); color: var(--pink);   background: rgba(255,45,120,0.06); }
.dot { width:6px; height:6px; border-radius:50%; background: currentColor; box-shadow: 0 0 5px currentColor; animation: blink 1.6s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

/* ── Code block ── */
.code {
    background: rgba(0,0,0,0.65);
    border: 1px solid rgba(0,212,255,0.18);
    border-left: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-family: var(--mono); font-size: 0.75rem; color: #90c8f0;
    overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    margin: 0.4rem 0;
}

/* ── Divider ── */
.divider {
    height:1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 1.1rem 0; opacity: 0.35;
}

/* ── History item ── */
.hist-item {
    background: rgba(0,8,24,0.7);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.45rem;
    font-size: 0.8rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.empty-title {
    font-family: var(--display); font-size: 1rem; letter-spacing: 0.18em;
    background: linear-gradient(135deg, var(--cyan), var(--purple), var(--pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.empty-sub { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); margin-top: 0.6rem; }

/* ── Step badge for onboarding ── */
.step-row { display: flex; align-items: flex-start; gap: 0.9rem; margin-bottom: 1rem; }
.step-num {
    font-family: var(--display); font-size: 0.65rem; font-weight: 700;
    min-width: 26px; height: 26px; border-radius: 50%;
    border: 1px solid var(--cyan); color: var(--cyan);
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,212,255,0.06); flex-shrink: 0; margin-top: 2px;
}
.step-text { font-family: var(--body); font-size: 0.82rem; color: #b0cce0; line-height: 1.5; }
.step-text strong { color: var(--text); }

/* ── Widget overrides ── */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: rgba(0,8,24,0.85) !important;
    border-color: rgba(0,200,255,0.28) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(0,90,170,0.55), rgba(70,0,130,0.55)) !important;
    border: 1px solid rgba(0,212,255,0.38) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: var(--display) !important; font-size: 0.68rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    padding: 0.55rem 1.1rem !important; transition: all 0.18s !important;
    width: 100%;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.3) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.28) !important;
    border-radius: 12px !important; background: rgba(0,8,24,0.55) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(0,120,80,0.45), rgba(0,80,40,0.45)) !important;
    border: 1px solid rgba(57,255,20,0.35) !important;
    color: var(--lime) !important;
    font-family: var(--display) !important; font-size: 0.68rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    border-radius: 8px !important; padding: 0.55rem 1.1rem !important;
    transition: all 0.18s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--lime) !important;
    box-shadow: 0 0 16px rgba(57,255,20,0.25) !important;
}
[data-baseweb="tab-list"] {
    background: rgba(0,8,24,0.65) !important;
    border-radius: 8px !important; padding: 3px !important; gap: 3px !important;
}
[data-baseweb="tab"] { font-family: var(--display) !important; font-size: 0.62rem !important; letter-spacing: 0.1em !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(0,140,255,0.22), rgba(110,0,190,0.22)) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
.stAlert { border-radius: 8px !important; font-family: var(--mono) !important; font-size: 0.8rem !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,.3); }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,.28); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "df_raw": None, "df_clean": None,
    "cleaning_history": [], "viz_code": [], "viz_figs": [],
    "model": None, "api_key_set": False, "api_key_source": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()

def safe_exec_cleaning(code: str, df: pd.DataFrame) -> pd.DataFrame:
    ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<clean>", "exec"), ns)  # noqa: S102
    return ns["df"]

def safe_exec_viz(code: str, df: pd.DataFrame):
    ns = {"df": df, "pd": pd, "np": np, "go": go, "px": px}
    exec(compile(code, "<viz>", "exec"), ns)  # noqa: S102
    return ns["create_figure"](df)

def ask_gemini(prompt: str) -> str:
    return st.session_state.model.generate_content(prompt).text

def health_report(df: pd.DataFrame) -> dict:
    tc = df.shape[0] * df.shape[1]
    m  = int(df.isnull().sum().sum())
    return {
        "rows": df.shape[0], "cols": df.shape[1], "missing": m,
        "missing_pct": round(m / tc * 100, 1) if tc else 0,
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

def badge(label, cls):
    return f'<span class="badge {cls}">{label}</span>'

def try_init_model(key: str) -> bool:
    """Try to initialise Gemini with the given key. Returns True on success."""
    if not GENAI_OK:
        return False
    try:
        genai.configure(api_key=key.strip())
        m = genai.GenerativeModel("gemini-1.5-flash")
        # Lightweight probe to validate the key
        m.generate_content("ping")
        st.session_state.model = m
        return True
    except Exception:
        st.session_state.model = None
        return False

VIZ_SCENE = dict(
    bgcolor="rgba(0,0,0,0)",
    xaxis=dict(backgroundcolor="rgba(0,10,30,.6)", gridcolor="rgba(0,180,255,.14)", showbackground=True, zerolinecolor="rgba(0,200,255,.18)", tickfont=dict(size=9)),
    yaxis=dict(backgroundcolor="rgba(5,0,20,.6)",  gridcolor="rgba(150,0,255,.14)",  showbackground=True, zerolinecolor="rgba(150,0,255,.18)", tickfont=dict(size=9)),
    zaxis=dict(backgroundcolor="rgba(0,5,20,.6)",  gridcolor="rgba(0,100,200,.14)",  showbackground=True, zerolinecolor="rgba(0,150,255,.18)", tickfont=dict(size=9)),
)

# ── Try to auto-load from Streamlit secrets ───────────────────────────────────
if not st.session_state.api_key_set:
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
    if secret_key:
        if try_init_model(secret_key):
            st.session_state.api_key_set   = True
            st.session_state.api_key_source = "secrets"

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">⬡ AetherVisuals v1.0</p>', unsafe_allow_html=True)

    # AI Status pill
    if st.session_state.api_key_set:
        src = f" · via {st.session_state.api_key_source}"
        st.markdown(f'<div class="pill pill-on"><span class="dot"></span>AI ONLINE{src}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill pill-off"><span class="dot"></span>AI OFFLINE</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── API Key section ──────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">🔑 Gemini API Key</p>', unsafe_allow_html=True)

    if st.session_state.api_key_set and st.session_state.api_key_source == "secrets":
        st.markdown('<div class="success-box">✓ Key pre-loaded from server secrets.<br>Paste below to override with your own.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box">'
            '🚀 <strong>Get a FREE key</strong><br>'
            'Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" '
            'style="color:#00d4ff;">aistudio.google.com</a><br>'
            'Sign in → <em>Create API Key</em> → copy &amp; paste below.<br>'
            '<span style="opacity:.7;font-size:.72rem;">Free tier · No credit card needed.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    user_key = st.text_input(
        "Paste your Gemini API key",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
        key="api_key_input",
    )
    activate = st.button("⚡ Activate Key", use_container_width=True)

    if activate:
        if not user_key.strip():
            st.error("Please paste a key first.")
        elif not GENAI_OK:
            st.error("google-generativeai package not installed.")
        else:
            with st.spinner("Validating key..."):
                ok = try_init_model(user_key.strip())
            if ok:
                st.session_state.api_key_set    = True
                st.session_state.api_key_source  = "user input"
                st.success("✅ Key activated — AI is online!")
                st.rerun()
            else:
                st.error("⛔ Invalid key or quota exceeded. Double-check and try again.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Pipeline status ──────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">📡 Pipeline</p>', unsafe_allow_html=True)
    for label, done in [
        ("AI Engine",    st.session_state.api_key_set),
        ("Data Loaded",  st.session_state.df_raw    is not None),
        ("Data Cleaned", st.session_state.df_clean  is not None),
        ("Viz Ready",    bool(st.session_state.viz_figs)),
    ]:
        icon = "🟢" if done else "⬜"
        st.markdown(f"{icon} `{label}`")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Quick-start guide ────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">📖 Quick Start</p>', unsafe_allow_html=True)
    for n, text in [
        ("1", "<strong>Paste your API key</strong> above and click Activate"),
        ("2", "<strong>Upload a CSV</strong> on the main page"),
        ("3", "<strong>Type a cleaning command</strong> in plain English"),
        ("4", "<strong>Generate 3D charts</strong> with one click"),
        ("5", "<strong>Download</strong> your clean data anytime"),
    ]:
        st.markdown(
            f'<div class="step-row"><div class="step-num">{n}</div>'
            f'<div class="step-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:var(--mono);font-size:.62rem;color:var(--muted);">Streamlit · Pandas · Gemini 1.5 Flash · Plotly</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
ai_pill = (
    '<span class="pill pill-on" style="font-size:.68rem;"><span class="dot"></span>AI ONLINE</span>'
    if st.session_state.api_key_set else
    '<span class="pill pill-off" style="font-size:.68rem;cursor:pointer;" '
    'title="Open the sidebar ← to add your API key">'
    '<span class="dot"></span>AI OFFLINE · Add key in sidebar ›</span>'
)
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:.9rem 0 1.3rem;border-bottom:1px solid var(--glow);margin-bottom:1.4rem;">
    <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2.2rem;">🌌</span>
        <div>
            <h1 class="title">AetherVisuals</h1>
            <p class="subtitle">AI-Powered · 3D Data Architect</p>
        </div>
    </div>
    {ai_pill}
</div>
""", unsafe_allow_html=True)

# ── Soft nudge if AI is offline ───────────────────────────────────────────────
if not st.session_state.api_key_set:
    st.markdown(
        '<div class="warn-box">⚠️ <strong>AI engine is offline.</strong> '
        'Open the sidebar (arrow on the left edge) → paste your free Gemini API key → click <em>Activate Key</em>. '
        'The app will then be fully functional.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — UPLOAD & HEALTH REPORT
# ─────────────────────────────────────────────────────────────────────────────
col_up, col_health = st.columns([1, 1.4], gap="medium")

with col_up:
    st.markdown('<p class="section-label">◈ Data Ingestion</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed",
        help="Upload any CSV — messy data is fine, that's what we're here for.",
    )
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.session_state.df_raw   = df_in.copy()
            st.session_state.df_clean = df_in.copy()
            st.session_state.cleaning_history = []
            st.session_state.viz_figs = []
            st.session_state.viz_code = []
            st.success(f"✅ **{uploaded.name}** loaded — {df_in.shape[0]:,} rows × {df_in.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.2rem 1rem;">'
            '<div style="font-size:2rem;margin-bottom:.5rem;">📂</div>'
            '<div style="font-family:var(--mono);font-size:.78rem;color:var(--muted);">'
            'Drag & drop a CSV here<br>'
            '<span style="font-size:.7rem;opacity:.6;">or click to browse</span></div></div>',
            unsafe_allow_html=True,
        )

with col_health:
    st.markdown('<p class="section-label">◈ Data Health Report</p>', unsafe_allow_html=True)
    if st.session_state.df_clean is not None:
        r = health_report(st.session_state.df_clean)
        mc = "b-ok" if r["missing"]==0 else ("b-warn" if r["missing_pct"]<10 else "b-bad")
        dc = "b-ok" if r["duplicates"]==0 else "b-warn"
        st.markdown(
            '<div class="badge-row">'
            + badge(f"📐 {r['rows']:,} rows",                         "b-info")
            + badge(f"🗂 {r['cols']} cols",                            "b-info")
            + badge(f"❓ {r['missing']} missing ({r['missing_pct']}%)", mc)
            + badge(f"♻️ {r['duplicates']} dupes",                     dc)
            + "</div>", unsafe_allow_html=True
        )
        miss_col = st.session_state.df_clean.isnull().sum()
        miss_col = miss_col[miss_col > 0]
        if not miss_col.empty:
            mdf = miss_col.reset_index()
            mdf.columns = ["Column", "Missing"]
            mdf["% Missing"] = (mdf["Missing"] / r["rows"] * 100).round(1)
            st.dataframe(mdf, use_container_width=True, hide_index=True, height=160)
        else:
            st.markdown('<span class="badge b-ok" style="margin-top:.6rem;display:inline-block;">✓ No missing values</span>', unsafe_allow_html=True)
        dtype_str = " · ".join([f"`{str(k).split('.')[-1]}` ×{v}" for k,v in r["dtypes"].items()])
        st.caption(f"Types: {dtype_str}")
    else:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.2rem 1rem;">'
            '<div style="font-family:var(--mono);font-size:.78rem;color:var(--muted);">'
            'Health report will appear here after upload.</div></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — DATA PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_clean is not None:
    with st.expander("🔬 Data Preview & Statistics", expanded=False):
        t1, t2, t3 = st.tabs(["📋 Table View", "📊 Statistics", "🔭 First 10 Rows"])
        with t1:
            st.dataframe(st.session_state.df_clean.head(50), use_container_width=True)
        with t2:
            st.dataframe(st.session_state.df_clean.describe(include="all").T, use_container_width=True)
        with t3:
            st.dataframe(st.session_state.df_clean.head(10), use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — JANITOR CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_clean is not None:
    st.markdown('<p class="section-label">◈ Janitor Console — Clean with Natural Language</p>', unsafe_allow_html=True)

    if not st.session_state.api_key_set:
        st.markdown('<div class="warn-box">⚠️ Add your API key in the sidebar to use the Janitor Console.</div>', unsafe_allow_html=True)
    
    col_j, col_h = st.columns([1.35, 1], gap="medium")

    with col_j:
        # Example command chips
        st.markdown(
            '<div style="font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.4rem;">'
            'Try: &nbsp;'
            '<span style="color:var(--cyan);cursor:default;">Remove duplicates</span> &nbsp;·&nbsp; '
            '<span style="color:var(--cyan);cursor:default;">Fill missing with median</span> &nbsp;·&nbsp; '
            '<span style="color:var(--cyan);cursor:default;">Remove outliers using IQR</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        cmd = st.text_area(
            "Command", height=110, label_visibility="collapsed",
            placeholder=(
                "Type your cleaning instruction in plain English…\n\n"
                "Examples:\n"
                "  · 'Drop all duplicate rows'\n"
                "  · 'Fill missing Price values with the column median'\n"
                "  · 'Remove outliers in the Sales column using IQR'\n"
                "  · 'Convert the Date column to datetime format'\n"
                "  · 'Create a new column Revenue = Price × Units'"
            ),
        )
        c1, c2 = st.columns(2)
        run   = c1.button("⚡ Execute", use_container_width=True, disabled=not st.session_state.api_key_set)
        reset = c2.button("🔄 Reset Data", use_container_width=True)

        if reset and st.session_state.df_raw is not None:
            st.session_state.df_clean = st.session_state.df_raw.copy()
            st.session_state.cleaning_history = []
            st.success("✅ Data reset to original.")
            st.rerun()

        if run and cmd.strip():
            with st.spinner("🤖 Writing Pandas code..."):
                prompt = f"""You are a senior data-cleaning engineer.
DataFrame info:
- Columns & dtypes: {st.session_state.df_clean.dtypes.to_dict()}
- Sample (5 rows):
{st.session_state.df_clean.head(5).to_string()}

User command: "{cmd}"

Rules:
1. Operate on variable `df` (Pandas DataFrame, already loaded).
2. Reassign `df = ...` or modify in-place so `df` holds the result.
3. Use only `pd` (pandas) and `np` (numpy) — do NOT write import statements.
4. Wrap risky operations in try/except so the script never crashes.
5. Return ONLY a ```python ... ``` code block — no explanation, no prose."""
                code = extract_code(ask_gemini(prompt))

            st.markdown('<p class="section-label" style="margin-top:.6rem;">Generated Code</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="code">{code}</div>', unsafe_allow_html=True)

            success, last_err, exec_code, attempt = False, None, code, 0
            while attempt < 2 and not success:
                try:
                    st.session_state.df_clean = safe_exec_cleaning(exec_code, st.session_state.df_clean)
                    st.session_state.cleaning_history.append({"cmd": cmd, "status": "✅"})
                    success = True
                except Exception as e:
                    last_err = str(e)
                    attempt += 1
                    if attempt < 2:
                        st.warning(f"⚠️ Error — self-healing... ({last_err[:80]})")
                        exec_code = extract_code(ask_gemini(
                            f"Fix this code:\n```python\n{exec_code}\n```\nError: {last_err}\n"
                            "Return ONLY the corrected ```python ... ``` block."
                        ))

            if success:
                df_s = st.session_state.df_clean
                st.success(f"✅ Done! Dataset is now **{df_s.shape[0]:,} rows × {df_s.shape[1]} columns**")
                st.rerun()
            else:
                st.error(f"❌ Could not apply command after 2 attempts.\n\n`{last_err}`")
                st.session_state.cleaning_history.append({"cmd": cmd, "status": f"❌ {last_err[:60]}"})

    with col_h:
        st.markdown('<p class="section-label">⟳ History</p>', unsafe_allow_html=True)
        hist = st.session_state.cleaning_history
        if hist:
            for i, e in enumerate(reversed(hist[-10:])):
                st.markdown(
                    f'<div class="hist-item">'
                    f'<span style="font-family:var(--mono);font-size:.68rem;color:var(--muted);">#{len(hist)-i} </span>'
                    f'{e["status"]} &nbsp;'
                    f'<span style="font-size:.78rem;">{e["cmd"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="font-family:var(--mono);font-size:.75rem;color:var(--muted);padding:.5rem 0;">No commands run yet.</div>', unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">⬇ Export</p>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=st.session_state.df_clean.to_csv(index=False).encode(),
            file_name="aether_cleaned.csv", mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — AI VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_clean is not None:
    st.markdown('<p class="section-label">◈ AI Visualizer — 3D Holographic Renderer</p>', unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 2.2], gap="medium")

    with col_ctrl:
        st.markdown(
            '<div class="glass">'
            '<div style="font-family:var(--mono);font-size:.76rem;color:#6a90b0;line-height:1.6;">'
            'Gemini analyses your dataset and proposes the <strong style="color:var(--cyan);">3 most insightful</strong> '
            '3D visualisations — automatically generated, interactive, and styled for depth.'
            '</div></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.api_key_set:
            st.markdown('<div class="warn-box">⚠️ Add your API key in the sidebar to generate visualisations.</div>', unsafe_allow_html=True)

        gen_btn = st.button("🚀 Generate 3D Visualizations", use_container_width=True, disabled=not st.session_state.api_key_set)

        if st.session_state.viz_figs:
            n = len(st.session_state.viz_figs)
            st.markdown(
                f'<div style="margin-top:.6rem;"><span class="badge b-ok">✓ {n} visualization{"s" if n>1 else ""} ready</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("🔁 Regenerate", use_container_width=True, disabled=not st.session_state.api_key_set):
                st.session_state.viz_figs = []
                st.session_state.viz_code = []
                st.rerun()

    with col_viz:
        if gen_btn:
            df = st.session_state.df_clean
            with st.spinner("🌌 Consulting Gemini for 3D blueprints..."):
                master = f"""You are a world-class data visualization expert and Python engineer.

Dataset:
- Columns & dtypes: {df.dtypes.to_dict()}
- Head (8 rows):
{df.head(8).to_string()}
- Statistics:
{df.describe(include="all").to_string()}

Task: Propose exactly 3 highly impactful 3D visualisations.
For EACH output a ```python ... ``` block that:
1. Defines `def create_figure(df):` → returns a plotly.graph_objects.Figure.
2. Uses only `go`, `pd`, `np` (pre-imported). NO import statements.
3. 3D axes only (scene with xaxis/yaxis/zaxis). Absolutely no 2D charts.
4. template="plotly_dark"
5. colorscale="Viridis" or colorscale="Electric"
6. paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
7. lighting=dict(ambient=0.4,diffuse=0.6,specular=0.8,roughness=0.5,fresnel=0.8) where applicable.
8. Descriptive figure title.
9. All data preprocessing inside the function. try/except for safety.

When generating Plotly code, always use 3D axes (x, y, z). Focus on depth and lighting. Return ONLY the python code block, no conversational text.

Output exactly 3 ```python ... ``` blocks. Nothing else — no prose, no numbering."""

                raw = ask_gemini(master)

            blocks = [b.strip() for b in re.findall(r"```(?:python)?\s*([\s\S]*?)```", raw) if "create_figure" in b]
            if not blocks:
                blocks = [raw.strip()]

            st.session_state.viz_code = blocks
            st.session_state.viz_figs = []
            prog = st.progress(0, text="Rendering visualizations...")

            for i, blk in enumerate(blocks[:3]):
                prog.progress((i+1)/3, text=f"Rendering chart {i+1} of 3...")
                fig, exec_blk, attempt = None, blk, 0
                while attempt < 2 and fig is None:
                    try:
                        fig = safe_exec_viz(exec_blk, df)
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Share Tech Mono, monospace", color="#90c8f0"),
                            margin=dict(l=10, r=10, t=45, b=10),
                            scene=VIZ_SCENE,
                        )
                        st.session_state.viz_figs.append(fig)
                    except Exception as e:
                        attempt += 1
                        if attempt < 2:
                            exec_blk = extract_code(ask_gemini(
                                f"Fix this Plotly code:\n```python\n{exec_blk}\n```\nError: {e}\n"
                                "Function must be `def create_figure(df):` → go.Figure.\n"
                                "Return ONLY the corrected ```python ... ``` block."
                            ))
                        else:
                            st.warning(f"⚠️ Chart {i+1} could not be rendered: {str(e)[:120]}")

            prog.empty()
            st.rerun()

    # ── Chart tabs ────────────────────────────────────────────────────────────
    if st.session_state.viz_figs:
        tabs = st.tabs([f"🌐 Chart {i+1}" for i in range(len(st.session_state.viz_figs))])
        for i, (tab, fig) in enumerate(zip(tabs, st.session_state.viz_figs)):
            with tab:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
                with st.expander(f"🧬 View source code — Chart {i+1}"):
                    if i < len(st.session_state.viz_code):
                        st.markdown(f'<div class="code">{st.session_state.viz_code[i]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🌌</div>
            <div class="empty-title">AWAITING DATA STREAM</div>
            <div class="empty-sub">Upload a CSV file above to initialise the pipeline</div>
            <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🧹</div>Clean with<br>natural language
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🤖</div>AI-generated<br>Pandas code
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🌐</div>Interactive<br>3D charts
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">⬇️</div>Export clean<br>CSV instantly
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
