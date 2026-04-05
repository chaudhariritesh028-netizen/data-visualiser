import streamlit as st
import pandas as pd
import numpy as np
import re

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AetherVisuals · AI 3D Data Architect",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
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
    --bg:      #020510;
    --glass:   rgba(5,15,40,0.75);
    --glow:    rgba(0,200,255,0.22);
    --cyan:    #00d4ff;
    --pink:    #ff2d78;
    --lime:    #39ff14;
    --purple:  #a78bfa;
    --text:    #e0f4ff;
    --muted:   #4a6a8a;
    --display: 'Orbitron', monospace;
    --mono:    'Share Tech Mono', monospace;
    --body:    'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 10%,  rgba(0,80,160,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 90%,  rgba(100,0,140,0.14) 0%, transparent 60%),
        repeating-linear-gradient(0deg,  transparent, transparent 39px, rgba(0,200,255,0.025) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,200,255,0.025) 40px);
    font-family: var(--body);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 3rem !important; max-width: 100% !important; }

/* ── Hide sidebar entirely ── */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }

/* ── API Key card ── */
.api-card {
    background: rgba(0,10,30,0.85);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 0 40px rgba(0,150,255,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 1rem;
}
.api-card-connected {
    border-color: rgba(57,255,20,0.35);
    background: rgba(0,20,10,0.8);
    box-shadow: 0 0 30px rgba(57,255,20,0.06);
}

/* ── Typography ── */
.title {
    font-family: var(--display);
    font-size: clamp(1.6rem, 2.8vw, 2.5rem);
    font-weight: 900;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 55%, var(--pink) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.1;
}
.subtitle {
    font-family: var(--mono);
    font-size: 0.7rem; color: var(--muted);
    letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.2rem;
}
.section-label {
    font-family: var(--display);
    font-size: 0.6rem; letter-spacing: 0.3em;
    text-transform: uppercase; color: var(--cyan);
    opacity: 0.85; margin: 0 0 0.5rem;
}

/* ── Pill ── */
.pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: var(--mono); font-size: 0.68rem;
    padding: 0.22rem 0.6rem; border-radius: 20px; letter-spacing: 0.08em;
}
.pill-on  { border: 1px solid rgba(57,255,20,0.5);  color: var(--lime);  background: rgba(57,255,20,0.06); }
.pill-off { border: 1px solid rgba(255,45,120,0.4); color: var(--pink);  background: rgba(255,45,120,0.06); }
.dot { width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 5px currentColor; animation:blink 1.6s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Glass card ── */
.glass {
    background: var(--glass);
    border: 1px solid var(--glow);
    border-radius: 12px; padding: 1.1rem 1.3rem;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 0 28px rgba(0,180,255,0.04), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 0.9rem;
}

/* ── Info / warn / success boxes ── */
.info-box  { background:rgba(0,212,255,0.05);  border:1px solid rgba(0,212,255,0.2);  border-left:3px solid var(--cyan);  border-radius:8px; padding:.75rem 1rem; font-family:var(--mono); font-size:.78rem; color:#80c8e8; margin-bottom:.7rem; }
.warn-box  { background:rgba(255,184,0,0.06);  border:1px solid rgba(255,184,0,0.25); border-left:3px solid #ffb800;      border-radius:8px; padding:.75rem 1rem; font-family:var(--mono); font-size:.78rem; color:#e8c870; margin-bottom:.7rem; }
.ok-box    { background:rgba(57,255,20,0.05);  border:1px solid rgba(57,255,20,0.2);  border-left:3px solid var(--lime);  border-radius:8px; padding:.75rem 1rem; font-family:var(--mono); font-size:.78rem; color:#90e870; margin-bottom:.7rem; }

/* ── Badges ── */
.badge-row { display:flex; gap:.6rem; flex-wrap:wrap; margin:.5rem 0; }
.badge { font-family:var(--mono); font-size:.74rem; padding:.3rem .7rem; border-radius:6px; border:1px solid; white-space:nowrap; }
.b-ok   { border-color:var(--lime);  color:var(--lime);  background:rgba(57,255,20,0.07); }
.b-warn { border-color:#ffb800;      color:#ffb800;      background:rgba(255,184,0,0.07); }
.b-bad  { border-color:var(--pink);  color:var(--pink);  background:rgba(255,45,120,0.07); }
.b-info { border-color:var(--cyan);  color:var(--cyan);  background:rgba(0,212,255,0.07); }

/* ── Code block ── */
.code {
    background:rgba(0,0,0,0.65); border:1px solid rgba(0,212,255,0.18);
    border-left:3px solid var(--cyan); border-radius:8px;
    padding:.85rem 1rem; font-family:var(--mono); font-size:.75rem; color:#90c8f0;
    overflow-x:auto; white-space:pre-wrap; word-break:break-word; margin:.4rem 0;
}

/* ── Divider ── */
.divider { height:1px; background:linear-gradient(90deg,transparent,var(--cyan),transparent); margin:1.1rem 0; opacity:.3; }

/* ── History item ── */
.hist-item {
    background:rgba(0,8,24,0.7); border:1px solid rgba(0,200,255,0.1);
    border-radius:8px; padding:.6rem .9rem; margin-bottom:.4rem; font-size:.8rem;
}

/* ── Empty state ── */
.empty-state { text-align:center; padding:4rem 2rem; }
.empty-title {
    font-family:var(--display); font-size:1rem; letter-spacing:.18em;
    background:linear-gradient(135deg,var(--cyan),var(--purple),var(--pink));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.empty-sub { font-family:var(--mono); font-size:.75rem; color:var(--muted); margin-top:.6rem; }

/* ── Widget overrides ── */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background:rgba(0,8,24,0.9) !important;
    border-color:rgba(0,200,255,0.28) !important;
    border-radius:8px !important; color:var(--text) !important;
    font-family:var(--mono) !important; font-size:.85rem !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color:var(--cyan) !important;
    box-shadow:0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stButton > button {
    background:linear-gradient(135deg,rgba(0,90,170,0.55),rgba(70,0,130,0.55)) !important;
    border:1px solid rgba(0,212,255,0.38) !important; border-radius:8px !important;
    color:var(--text) !important; font-family:var(--display) !important;
    font-size:.68rem !important; letter-spacing:.14em !important;
    text-transform:uppercase !important; padding:.55rem 1.1rem !important;
    transition:all .18s !important; width:100%;
}
.stButton > button:hover {
    border-color:var(--cyan) !important;
    box-shadow:0 0 16px rgba(0,212,255,0.3) !important;
    transform:translateY(-1px) !important;
}
.stButton > button:disabled {
    opacity:.35 !important; cursor:not-allowed !important;
    transform:none !important; box-shadow:none !important;
}
[data-testid="stFileUploader"] {
    border:2px dashed rgba(0,212,255,0.28) !important;
    border-radius:12px !important; background:rgba(0,8,24,0.55) !important;
    transition:border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--cyan) !important; }
.stDownloadButton > button {
    background:linear-gradient(135deg,rgba(0,120,80,0.45),rgba(0,80,40,0.45)) !important;
    border:1px solid rgba(57,255,20,0.35) !important; color:var(--lime) !important;
    font-family:var(--display) !important; font-size:.68rem !important;
    letter-spacing:.14em !important; text-transform:uppercase !important;
    border-radius:8px !important; padding:.55rem 1.1rem !important;
    transition:all .18s !important;
}
.stDownloadButton > button:hover {
    border-color:var(--lime) !important;
    box-shadow:0 0 16px rgba(57,255,20,0.25) !important;
}
[data-baseweb="tab-list"] {
    background:rgba(0,8,24,0.65) !important;
    border-radius:8px !important; padding:3px !important; gap:3px !important;
}
[data-baseweb="tab"] { font-family:var(--display) !important; font-size:.62rem !important; letter-spacing:.1em !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background:linear-gradient(135deg,rgba(0,140,255,0.22),rgba(110,0,190,0.22)) !important;
    border-bottom:2px solid var(--cyan) !important;
}
.stAlert { border-radius:8px !important; font-family:var(--mono) !important; font-size:.8rem !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(0,0,0,.3); }
::-webkit-scrollbar-thumb { background:rgba(0,180,255,.28); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "df_raw": None, "df_clean": None,
    "cleaning_history": [], "viz_code": [], "viz_figs": [],
    "model": None, "ai_ready": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_code(text):
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()

def safe_exec_cleaning(code, df):
    ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<clean>", "exec"), ns)  # noqa: S102
    return ns["df"]

def safe_exec_viz(code, df):
    ns = {"df": df, "pd": pd, "np": np, "go": go, "px": px}
    exec(compile(code, "<viz>", "exec"), ns)  # noqa: S102
    return ns["create_figure"](df)

def ask_gemini(prompt):
    return st.session_state.model.generate_content(prompt).text

def health_report(df):
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

def try_init_model(key):
    if not GENAI_OK or not key.strip():
        return False, "google-generativeai package missing"
    try:
        genai.configure(api_key=key.strip())
        m = genai.GenerativeModel("gemini-1.5-flash")
        m.generate_content("hi")          # quick validation ping
        st.session_state.model    = m
        st.session_state.ai_ready = True
        return True, ""
    except Exception as e:
        st.session_state.model    = None
        st.session_state.ai_ready = False
        return False, str(e)

VIZ_SCENE = dict(
    bgcolor="rgba(0,0,0,0)",
    xaxis=dict(backgroundcolor="rgba(0,10,30,.6)",  gridcolor="rgba(0,180,255,.14)",  showbackground=True, zerolinecolor="rgba(0,200,255,.18)", tickfont=dict(size=9)),
    yaxis=dict(backgroundcolor="rgba(5,0,20,.6)",   gridcolor="rgba(150,0,255,.14)",  showbackground=True, zerolinecolor="rgba(150,0,255,.18)", tickfont=dict(size=9)),
    zaxis=dict(backgroundcolor="rgba(0,5,20,.6)",   gridcolor="rgba(0,100,200,.14)",  showbackground=True, zerolinecolor="rgba(0,150,255,.18)", tickfont=dict(size=9)),
)

# ── Auto-load from Streamlit secrets ─────────────────────────────────────────
if not st.session_state.ai_ready:
    secret = st.secrets.get("GEMINI_API_KEY", "")
    if secret:
        try_init_model(secret)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
ai_pill = (
    '<span class="pill pill-on"><span class="dot"></span>AI ONLINE</span>'
    if st.session_state.ai_ready else
    '<span class="pill pill-off"><span class="dot"></span>AI OFFLINE</span>'
)
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:.8rem 0 1.2rem;border-bottom:1px solid rgba(0,200,255,0.2);
            margin-bottom:1.4rem;">
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

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — API KEY  (always visible on main page)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">◈ Step 1 — Connect AI Engine</p>', unsafe_allow_html=True)

if st.session_state.ai_ready:
    # Connected state — show compact confirmation, allow override
    with st.expander("✅ AI Engine Connected — click to change key", expanded=False):
        new_key = st.text_input(
            "New API Key", type="password",
            placeholder="Paste a different key to override…",
            label_visibility="collapsed",
        )
        if st.button("🔄 Switch Key", use_container_width=True):
            if new_key.strip():
                ok, err = try_init_model(new_key.strip())
                if ok:
                    st.success("✅ Key switched!")
                    st.rerun()
                else:
                    st.error(f"⛔ {err}")
else:
    # Disconnected state — prominent key entry
    col_key, col_help = st.columns([1.6, 1], gap="medium")

    with col_key:
        st.markdown(
            '<div class="api-card">'
            '<div style="font-family:var(--display);font-size:.65rem;letter-spacing:.25em;'
            'text-transform:uppercase;color:var(--cyan);margin-bottom:.8rem;">🔑 Gemini API Key</div>'
            '<div style="font-family:var(--mono);font-size:.78rem;color:#6a90b0;margin-bottom:.9rem;">'
            'Paste your free Google Gemini key below to activate the AI engine.<br>'
            '<span style="opacity:.6;font-size:.72rem;">Your key is never stored — session only.</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="AIzaSy...",
            label_visibility="collapsed",
            key="api_key_field",
        )
        activate_btn = st.button("⚡ Activate AI Engine", use_container_width=True)

        if activate_btn:
            if not api_key_input.strip():
                st.error("⛔ Please paste your API key first.")
            else:
                with st.spinner("🔌 Connecting to Gemini..."):
                    ok, err = try_init_model(api_key_input.strip())
                if ok:
                    st.success("✅ Connected! AI engine is online.")
                    st.rerun()
                else:
                    st.error(f"⛔ Could not connect: {err}\n\nDouble-check your key and try again.")

    with col_help:
        st.markdown(
            '<div class="api-card" style="height:100%;">'
            '<div style="font-family:var(--display);font-size:.6rem;letter-spacing:.25em;'
            'text-transform:uppercase;color:var(--cyan);margin-bottom:.8rem;">📖 How to get a free key</div>'
            '<div style="font-family:var(--mono);font-size:.78rem;color:#6a90b0;line-height:1.9;">'
            '1️⃣ &nbsp;Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" '
            'style="color:var(--cyan);text-decoration:none;">aistudio.google.com</a><br>'
            '2️⃣ &nbsp;Sign in with Google<br>'
            '3️⃣ &nbsp;Click <strong style="color:var(--text);">Create API Key</strong><br>'
            '4️⃣ &nbsp;Copy &amp; paste it here<br>'
            '<span style="opacity:.5;font-size:.7rem;margin-top:.4rem;display:block;">'
            '✓ Free &nbsp;✓ No credit card &nbsp;✓ Instant</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — UPLOAD & HEALTH REPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">◈ Step 2 — Upload Your Data</p>', unsafe_allow_html=True)

col_up, col_health = st.columns([1, 1.4], gap="medium")

with col_up:
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        label_visibility="collapsed",
        help="Any CSV file works — messy data is fine.",
    )
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.session_state.df_raw   = df_in.copy()
            st.session_state.df_clean = df_in.copy()
            st.session_state.cleaning_history = []
            st.session_state.viz_figs = []
            st.session_state.viz_code = []
            st.success(f"✅ **{uploaded.name}** — {df_in.shape[0]:,} rows × {df_in.shape[1]} cols")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.2rem 1rem;">'
            '<div style="font-size:2rem;margin-bottom:.5rem;">📂</div>'
            '<div style="font-family:var(--mono);font-size:.78rem;color:var(--muted);">'
            'Drag & drop a CSV here<br>'
            '<span style="font-size:.7rem;opacity:.5;">or click to browse your files</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

with col_health:
    if st.session_state.df_clean is not None:
        r   = health_report(st.session_state.df_clean)
        mc  = "b-ok" if r["missing"]==0 else ("b-warn" if r["missing_pct"]<10 else "b-bad")
        dc  = "b-ok" if r["duplicates"]==0 else "b-warn"
        st.markdown(
            '<div class="badge-row">'
            + badge(f"📐 {r['rows']:,} rows",                          "b-info")
            + badge(f"🗂 {r['cols']} cols",                             "b-info")
            + badge(f"❓ {r['missing']} missing ({r['missing_pct']}%)", mc)
            + badge(f"♻️ {r['duplicates']} dupes",                      dc)
            + "</div>", unsafe_allow_html=True
        )
        mc2 = st.session_state.df_clean.isnull().sum()
        mc2 = mc2[mc2 > 0]
        if not mc2.empty:
            mdf = mc2.reset_index()
            mdf.columns = ["Column", "Missing"]
            mdf["% Missing"] = (mdf["Missing"] / r["rows"] * 100).round(1)
            st.dataframe(mdf, use_container_width=True, hide_index=True, height=160)
        else:
            st.markdown('<span class="badge b-ok" style="display:inline-block;margin-top:.5rem;">✓ No missing values</span>', unsafe_allow_html=True)
        dtype_str = " · ".join([f"`{str(k).split('.')[-1]}` ×{v}" for k,v in r["dtypes"].items()])
        st.caption(f"Column types: {dtype_str}")
    else:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.2rem 1rem;">'
            '<div style="font-family:var(--mono);font-size:.78rem;color:var(--muted);">'
            'Health report appears here after upload.</div></div>',
            unsafe_allow_html=True,
        )

# Data preview expander
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
#  STEP 3 — JANITOR CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">◈ Step 3 — Clean Your Data</p>', unsafe_allow_html=True)

if st.session_state.df_clean is None:
    st.markdown('<div class="warn-box">⬆️ Upload a CSV file in Step 2 first.</div>', unsafe_allow_html=True)
elif not st.session_state.ai_ready:
    st.markdown('<div class="warn-box">🔑 Connect your API key in Step 1 to use the Janitor Console.</div>', unsafe_allow_html=True)
else:
    col_j, col_h = st.columns([1.4, 1], gap="medium")

    with col_j:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.4rem;">'
            '💡 Try: &nbsp;'
            '<span style="color:var(--cyan);">Remove duplicate rows</span>'
            ' &nbsp;·&nbsp; <span style="color:var(--cyan);">Fill missing values with median</span>'
            ' &nbsp;·&nbsp; <span style="color:var(--cyan);">Remove outliers using IQR</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        cmd = st.text_area(
            "Command", height=110, label_visibility="collapsed",
            placeholder=(
                "Type a cleaning instruction in plain English…\n\n"
                "Examples:\n"
                "  · 'Drop all duplicate rows'\n"
                "  · 'Fill missing Price values with the column median'\n"
                "  · 'Remove outliers in Sales column using IQR'\n"
                "  · 'Convert the Date column to datetime format'\n"
                "  · 'Create a new column Revenue = Price × Units'"
            ),
        )
        c1, c2 = st.columns(2)
        run   = c1.button("⚡ Execute",    use_container_width=True)
        reset = c2.button("🔄 Reset Data", use_container_width=True)

        if reset:
            st.session_state.df_clean         = st.session_state.df_raw.copy()
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
1. Operate on variable `df` (already a Pandas DataFrame).
2. Reassign `df = ...` or modify in-place so `df` holds the result.
3. Use only `pd` (pandas) and `np` (numpy) — no import statements.
4. Wrap risky ops in try/except.
5. Return ONLY a ```python ... ``` code block — no prose."""
                code = extract_code(ask_gemini(prompt))

            st.markdown('<p class="section-label" style="margin-top:.5rem;">Generated Code</p>', unsafe_allow_html=True)
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
                st.error(f"❌ Could not apply after 2 attempts.\n\n`{last_err}`")
                st.session_state.cleaning_history.append({"cmd": cmd, "status": f"❌ {last_err[:60]}"})

    with col_h:
        st.markdown('<p class="section-label">⟳ History</p>', unsafe_allow_html=True)
        hist = st.session_state.cleaning_history
        if hist:
            for i, e in enumerate(reversed(hist[-10:])):
                st.markdown(
                    f'<div class="hist-item">'
                    f'<span style="font-family:var(--mono);font-size:.68rem;color:var(--muted);">#{len(hist)-i} </span>'
                    f'{e["status"]} &nbsp;<span style="font-size:.78rem;">{e["cmd"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="font-family:var(--mono);font-size:.75rem;color:var(--muted);">No commands yet.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">⬇ Export</p>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=st.session_state.df_clean.to_csv(index=False).encode(),
            file_name="aether_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — AI VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">◈ Step 4 — Generate 3D Visualizations</p>', unsafe_allow_html=True)

if st.session_state.df_clean is None:
    st.markdown('<div class="warn-box">⬆️ Upload a CSV file in Step 2 first.</div>', unsafe_allow_html=True)
elif not st.session_state.ai_ready:
    st.markdown('<div class="warn-box">🔑 Connect your API key in Step 1 to generate visualisations.</div>', unsafe_allow_html=True)
else:
    col_ctrl, col_viz = st.columns([1, 2.2], gap="medium")

    with col_ctrl:
        st.markdown(
            '<div class="glass">'
            '<div style="font-family:var(--mono);font-size:.76rem;color:#6a90b0;line-height:1.7;">'
            'Gemini analyses your dataset and builds the '
            '<strong style="color:var(--cyan);">3 most insightful</strong> '
            '3D visualisations — automatically generated, interactive, '
            'and styled with depth &amp; lighting.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        gen_btn = st.button("🚀 Generate 3D Visualizations", use_container_width=True)

        if st.session_state.viz_figs:
            n = len(st.session_state.viz_figs)
            st.markdown(
                f'<div style="margin:.5rem 0;"><span class="badge b-ok">'
                f'✓ {n} chart{"s" if n>1 else ""} ready</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("🔁 Regenerate Charts", use_container_width=True):
                st.session_state.viz_figs = []
                st.session_state.viz_code = []
                st.rerun()

    with col_viz:
        if gen_btn:
            df = st.session_state.df_clean
            with st.spinner("🌌 Asking Gemini for 3D visualization blueprints..."):
                master = f"""You are a world-class data visualization expert and Python engineer.

Dataset:
- Columns & dtypes: {df.dtypes.to_dict()}
- Head (8 rows):
{df.head(8).to_string()}
- Statistics:
{df.describe(include="all").to_string()}

Task: Propose exactly 3 highly impactful 3D visualisations.
For EACH output a ```python ... ``` block that:
1. Defines `def create_figure(df):` returning a plotly.graph_objects.Figure.
2. Uses only `go`, `pd`, `np` — already imported, NO import statements.
3. 3D axes only (scene with xaxis/yaxis/zaxis). Absolutely no 2D charts.
4. template="plotly_dark"
5. colorscale="Viridis" or colorscale="Electric"
6. paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
7. lighting=dict(ambient=0.4,diffuse=0.6,specular=0.8,roughness=0.5,fresnel=0.8) where applicable.
8. Descriptive figure title.
9. All preprocessing inside the function, try/except for safety.

When generating Plotly code, always use 3D axes (x, y, z). Return ONLY 3 ```python ... ``` blocks, nothing else."""

                raw = ask_gemini(master)

            blocks = [b.strip() for b in re.findall(r"```(?:python)?\s*([\s\S]*?)```", raw) if "create_figure" in b]
            if not blocks:
                blocks = [raw.strip()]

            st.session_state.viz_code = blocks
            st.session_state.viz_figs = []
            prog = st.progress(0, text="Rendering chart 1 of 3...")

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
                            st.warning(f"⚠️ Chart {i+1} could not render: {str(e)[:100]}")
            prog.empty()
            st.rerun()

    # Charts
    if st.session_state.viz_figs:
        tabs = st.tabs([f"🌐 Chart {i+1}" for i in range(len(st.session_state.viz_figs))])
        for i, (tab, fig) in enumerate(zip(tabs, st.session_state.viz_figs)):
            with tab:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
                with st.expander(f"🧬 Source Code — Chart {i+1}"):
                    if i < len(st.session_state.viz_code):
                        st.markdown(f'<div class="code">{st.session_state.viz_code[i]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df_raw is None and not st.session_state.ai_ready:
    st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem;margin-bottom:.8rem;">🌌</div>
            <div class="empty-title">GET STARTED IN 2 STEPS</div>
            <div class="empty-sub">
                1 · Paste your free Gemini API key above and click Activate<br>
                2 · Upload a CSV file — AetherVisuals handles the rest
            </div>
            <div style="margin-top:2.5rem;display:flex;justify-content:center;gap:2.5rem;flex-wrap:wrap;">
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🧹</div>Clean with<br>plain English
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🤖</div>AI writes<br>the code
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">🌐</div>Interactive<br>3D charts
                </div>
                <div style="font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:center;">
                    <div style="font-size:1.4rem;margin-bottom:.3rem;">⬇️</div>Download<br>clean CSV
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
