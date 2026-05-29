"""
ECOVERGE — Biodegradable Packaging Analytics Dashboard
======================================================
A complete Business Data Analytics dashboard for ECOVERGE, a proposed B2B
biodegradable-packaging company in India. 16 analytics sections from
descriptive KPIs through predictive ML, association rules, forecasting,
recommender, sentiment, network analysis, ethics, and lead scoring.

Author: Business Data Analytics assessment project.
"""

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve,
                             mean_squared_error, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")

# Optional libraries — graceful degradation if missing on Streamlit Cloud
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except Exception:
    HAS_VADER = False


# ===========================================================================
# PAGE CONFIG + GLOBAL CSS  (deep forest / earthy palette for a B2B
# sustainability brand — distinct from generic AI-dashboard aesthetics)
# ===========================================================================

st.set_page_config(
    page_title="ECOVERGE — Biodegradable Packaging Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette anchors
ECO_DEEP = "#0F3D2E"     # deep forest
ECO_LEAF = "#2C7A4B"     # leaf green
ECO_MOSS = "#5A8A3A"     # moss
ECO_KRAFT = "#C7A87C"    # kraft paper accent
ECO_CLAY = "#B85C38"     # clay/terracotta
ECO_SAND = "#F1ECDF"     # sand background
ECO_INK = "#1A2421"      # almost-black ink
PLOTLY_TEMPLATE = "simple_white"

# Categorical sequence for charts
ECO_SEQ = [ECO_LEAF, ECO_KRAFT, ECO_CLAY, ECO_DEEP, ECO_MOSS,
           "#7BA05B", "#9C6B3B", "#3D6B4A", "#D4A574", "#4A7A5C"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@300;400;500;600&display=swap');

:root {{
  --eco-deep:{ECO_DEEP};
  --eco-leaf:{ECO_LEAF};
  --eco-moss:{ECO_MOSS};
  --eco-kraft:{ECO_KRAFT};
  --eco-clay:{ECO_CLAY};
  --eco-sand:{ECO_SAND};
  --eco-ink:{ECO_INK};
}}

html, body, [class*="css"] {{
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--eco-ink);
}}

h1, h2, h3, h4 {{
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  color: var(--eco-deep);
  letter-spacing: -0.01em;
}}

h1 {{ font-size: 2.4rem; line-height:1.15; }}
h2 {{ font-size: 1.55rem; margin-top: 1.4rem; }}
h3 {{ font-size: 1.2rem; }}

.eco-hero {{
  background: linear-gradient(135deg, var(--eco-deep) 0%, var(--eco-leaf) 100%);
  padding: 2.2rem 2rem;
  border-radius: 16px;
  color: #F8F5EC;
  margin-bottom: 1.4rem;
  box-shadow: 0 8px 32px rgba(15,61,46,0.18);
}}
.eco-hero h1 {{ color:#F8F5EC; margin:0 0 .35rem 0; font-size: 2.5rem; }}
.eco-hero p {{ color:#E6E0CC; opacity:0.92; font-size: 1.02rem; margin:0; }}
.eco-hero .tag {{
  display:inline-block; padding: .25rem .65rem; border-radius: 999px;
  background: rgba(199,168,124,0.20); color: var(--eco-kraft);
  font-size:.78rem; font-weight:500; letter-spacing:.04em;
  margin-bottom: .8rem; text-transform: uppercase;
}}

.kpi-card {{
  background: #FFFFFF;
  border: 1px solid #E5E0CF;
  border-left: 4px solid var(--eco-leaf);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin-bottom: .6rem;
  transition: all .18s ease;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(15,61,46,.10); }}
.kpi-card.alt {{ border-left-color: var(--eco-kraft); }}
.kpi-card.warn {{ border-left-color: var(--eco-clay); }}
.kpi-value {{ font-family:'Fraunces', serif; font-size:1.65rem; font-weight:700; color: var(--eco-deep); line-height:1.0; }}
.kpi-label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:#5C6A65; margin-top:.3rem; }}

.biz-note {{
  background: var(--eco-sand);
  border-radius: 10px;
  padding: 1rem 1.15rem;
  border-left: 3px solid var(--eco-clay);
  margin: .75rem 0 1.1rem 0;
  font-size: .95rem;
  line-height: 1.55;
}}
.biz-note b {{ color: var(--eco-deep); }}
.biz-note .label {{
  display:block; font-size:.7rem; letter-spacing:.10em;
  text-transform:uppercase; color: var(--eco-clay); font-weight:600;
  margin-bottom:.35rem;
}}

.section-intro {{
  font-size: 1rem; color:#4A554F; line-height:1.6;
  border-left: 3px solid var(--eco-kraft);
  padding-left: 1rem; margin: .4rem 0 1.4rem 0;
}}

.persona-card {{
  background: #FFFFFF; border:1px solid #E5E0CF; border-radius: 12px;
  padding: 1.1rem 1.2rem; margin-bottom: .8rem;
  border-top: 4px solid var(--eco-leaf);
}}
.persona-card h4 {{ margin:0 0 .4rem 0; }}
.persona-tag {{
  display:inline-block; background: var(--eco-sand); color: var(--eco-deep);
  padding: .2rem .55rem; border-radius:6px; font-size:.75rem; margin-right:.3rem;
  font-weight:500;
}}

.lead-hot   {{ background: var(--eco-leaf); color:white; padding:.3rem .7rem;
              border-radius:6px; font-weight:600; display:inline-block; }}
.lead-warm  {{ background: var(--eco-kraft); color:var(--eco-ink); padding:.3rem .7rem;
              border-radius:6px; font-weight:600; display:inline-block; }}
.lead-cold  {{ background: #C5C2B3; color:var(--eco-ink); padding:.3rem .7rem;
              border-radius:6px; font-weight:600; display:inline-block; }}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #F4EFE0 0%, #ECE5D2 100%);
}}
[data-testid="stSidebar"] h2 {{ color: var(--eco-deep); }}

div.stButton > button {{
  background: var(--eco-leaf); color: white; border: none;
  border-radius: 8px; padding: .5rem 1.2rem; font-weight: 500;
}}
div.stButton > button:hover {{ background: var(--eco-deep); }}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HELPER COMPONENTS
# ===========================================================================

def kpi_card(container, value, label, alt=False, warn=False):
    klass = "kpi-card" + (" alt" if alt else "") + (" warn" if warn else "")
    container.markdown(
        f'<div class="{klass}"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def biz_note(what, why, action):
    st.markdown(
        f'<div class="biz-note">'
        f'<span class="label">Business interpretation</span>'
        f'<b>What it shows:</b> {what}<br>'
        f'<b>Why it matters:</b> {why}<br>'
        f'<b>What ECOVERGE should do:</b> {action}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_intro(text):
    st.markdown(f'<div class="section-intro">{text}</div>', unsafe_allow_html=True)


def hero(title, sub, tag="ECOVERGE ANALYTICS"):
    st.markdown(
        f'<div class="eco-hero"><span class="tag">{tag}</span>'
        f'<h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )


# Columns that would leak the target — exclude from any ML feature set
LEAK_COLS = [
    "respondent_id", "feedback_text",
    "likely_adopter", "adoption_intention",
    "expected_biodegradable_budget", "willingness_to_pay_premium_pct",
    "lead_priority_score", "adoption_readiness_score",
    "customer_value_segment", "adoption_risk_category",
]


# ===========================================================================
# DATA LOAD + CACHED MODEL FUNCTIONS
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_data(path="ecoverge_data.csv"):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def build_feature_matrix(df, exclude_extra=None):
    """One-hot encoded numeric feature matrix, leakage columns dropped."""
    exclude = set(LEAK_COLS) | set(exclude_extra or [])
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    return X


@st.cache_resource(show_spinner=False)
def train_classifiers(df):
    """Train 5 classifiers on the FULL dataset for stability."""
    X = build_feature_matrix(df)
    y = (df["likely_adopter"] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=240, max_depth=12,
                                                       random_state=42, n_jobs=-1), False),
        "KNN":                 (KNeighborsClassifier(n_neighbors=15), True),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=200,
                                                            max_depth=4, random_state=42), False),
    }

    rows, fitted = [], {}
    for name, (model, needs_scale) in models.items():
        Xtr_in, Xte_in = (Xtr_s, Xte_s) if needs_scale else (Xtr, Xte)
        model.fit(Xtr_in, ytr)
        pred = model.predict(Xte_in)
        proba = (model.predict_proba(Xte_in)[:, 1]
                 if hasattr(model, "predict_proba") else pred)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(yte, pred), 3),
            "Precision": round(precision_score(yte, pred, zero_division=0), 3),
            "Recall": round(recall_score(yte, pred), 3),
            "F1": round(f1_score(yte, pred), 3),
            "AUC": round(roc_auc_score(yte, proba), 3),
        })
        fitted[name] = {"model": model, "scale": needs_scale,
                        "Xte": Xte_in, "yte": yte, "proba": proba, "pred": pred}

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return {"results": results, "fitted": fitted, "scaler": sc,
            "cols": list(X.columns), "X": X, "y": y}


@st.cache_resource(show_spinner=False)
def train_regressors(df):
    """Regression on log(expected_biodegradable_budget)."""
    extra = ["expected_biodegradable_budget", "monthly_packaging_spend",
             "willingness_to_pay_premium_pct"]
    X = build_feature_matrix(df, exclude_extra=extra)
    y = np.log1p(df["expected_biodegradable_budget"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Linear Regression":    LinearRegression(),
        "Decision Tree":        DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":        RandomForestRegressor(n_estimators=240, max_depth=14,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=240,
                                                            max_depth=4, random_state=42),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        # back-transform for INR-scale metrics
        pred_inr = np.expm1(pred); actual_inr = np.expm1(yte)
        rows.append({
            "Model": name,
            "MAE": int(mean_absolute_error(actual_inr, pred_inr)),
            "RMSE": int(np.sqrt(mean_squared_error(actual_inr, pred_inr))),
            "R²": round(r2_score(yte, pred), 3),
        })
        fitted[name] = m

    results = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    return {"results": results, "fitted": fitted,
            "cols": list(X.columns), "X": X, "y": y, "Xte": Xte, "yte": yte}


@st.cache_data(show_spinner=False)
def run_clustering(df, k=4):
    feat_cols = [
        "sustainability_mindset_score", "supplier_pain_score",
        "switching_barrier_score", "trust_builder_score",
        "packaging_demand_score", "commercial_value_score",
        "adoption_readiness_score", "customer_visibility_score",
        "monthly_packaging_spend", "n_employees",
    ]
    X = df[feat_cols].copy()
    X["monthly_packaging_spend"] = np.log1p(X["monthly_packaging_spend"])
    X["n_employees"] = np.log1p(X["n_employees"])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    inertias, sils = [], []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))

    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    labels = km.labels_
    return {"X": X, "Xs": Xs, "labels": labels, "k_range": list(range(2, 9)),
            "inertias": inertias, "sils": sils, "feat_cols": feat_cols}


@st.cache_data(show_spinner=False)
def mine_rules(df, min_support=0.10, min_lift=1.05):
    if not HAS_MLXTEND:
        return None
    prefs = ["pref_food_containers", "pref_bowls", "pref_cups", "pref_carry_bags",
             "pref_paper_mailers", "pref_corrugated_boxes", "pref_meal_trays",
             "pref_protective_packaging", "pref_premium_branded"]
    nice = {"pref_food_containers": "Food Containers", "pref_bowls": "Bowls",
            "pref_cups": "Cups", "pref_carry_bags": "Carry Bags",
            "pref_paper_mailers": "Paper Mailers", "pref_corrugated_boxes": "Corrugated Boxes",
            "pref_meal_trays": "Meal Trays", "pref_protective_packaging": "Protective Packaging",
            "pref_premium_branded": "Premium Branded"}
    basket = (df[prefs] == "Yes").rename(columns=nice)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()
    try:
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    except TypeError:
        # mlxtend 0.23.x signature requires num_itemsets (number of transactions)
        rules = association_rules(freq, num_itemsets=len(basket),
                                  metric="lift", min_threshold=min_lift)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules


@st.cache_data(show_spinner=False)
def simulate_sales(months=24, base=180, growth=0.025, seasonal_amp=0.18, seed=7):
    """Simulated ECOVERGE monthly sales (units in '000s) over 24 months."""
    rs = np.random.default_rng(seed)
    t = np.arange(months)
    trend = base * (1 + growth) ** t
    season = 1 + seasonal_amp * np.sin(2 * np.pi * (t % 12) / 12 + 0.5)
    noise = rs.normal(0, 0.05, months)
    sales = (trend * season * (1 + noise)).round().astype(int)
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=months, freq="MS")
    return pd.DataFrame({"month": dates, "sales_k_units": sales})


@st.cache_data(show_spinner=False)
def build_referral_graph(df, max_nodes=80):
    """Synthetic referral network — connect high-influence businesses to peers."""
    top = df.nlargest(max_nodes, "lead_priority_score").reset_index(drop=True)
    G = nx.DiGraph()
    rs = np.random.default_rng(42)
    for _, row in top.iterrows():
        G.add_node(row["respondent_id"], biz=row["business_type"],
                   score=float(row["lead_priority_score"]),
                   referrals=int(row["referral_count"]))
    ids = top["respondent_id"].tolist()
    for i, row in top.iterrows():
        n_out = int(row["referral_count"] * 0.5)
        if n_out <= 0:
            continue
        # connect to lower-priority peers
        candidates = [x for j, x in enumerate(ids) if j != i]
        targets = rs.choice(candidates, size=min(n_out, len(candidates)), replace=False)
        for t in targets:
            G.add_edge(row["respondent_id"], t)
    return G


# ===========================================================================
# DATA LOAD + GLOBAL BOUNDS
# ===========================================================================
df_raw = load_data()

# Realistic bounds for the regression target — clamp predictions so the
# linear model never reports a nonsensical (negative or huge) budget.
BUDGET_LO = float(df_raw["expected_biodegradable_budget"].quantile(0.01))
BUDGET_HI = float(df_raw["expected_biodegradable_budget"].quantile(0.99))


def clamp_budget(x):
    return float(np.clip(x, BUDGET_LO, BUDGET_HI))


# ===========================================================================
# SIDEBAR — navigation + global filters
# ===========================================================================

st.sidebar.markdown("## 🌿 ECOVERGE Analytics")
st.sidebar.caption("Biodegradable packaging for India's B2B market")

SECTIONS = [
    "🏠 Home & Objectives",
    "🧬 Feature Engineering",
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Classification Models",
    "🌳 Decision Tree",
    "👥 Clustering & Personas",
    "🔗 Association Rules",
    "💰 Regression (Budget Prediction)",
    "📈 Forecasting",
    "🎁 Recommender System",
    "💬 Text Mining & Sentiment",
    "🌐 Referral Network",
    "⚖️ Ethics & Sustainability",
    "🧭 Business Recommendations",
    "🆕 New Customer / Lead Prediction",
]
page = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Global filters")

biz_filter = st.sidebar.multiselect(
    "Business type", sorted(df_raw["business_type"].unique()),
    default=sorted(df_raw["business_type"].unique()))
tier_filter = st.sidebar.multiselect(
    "City tier", sorted(df_raw["city_tier"].unique()),
    default=sorted(df_raw["city_tier"].unique()))
rev_filter = st.sidebar.multiselect(
    "Revenue band", df_raw["annual_revenue_band"].unique().tolist(),
    default=df_raw["annual_revenue_band"].unique().tolist())
adopt_filter = st.sidebar.multiselect(
    "Adoption intention (filter)",
    sorted(df_raw["likely_adopter"].unique()),
    default=sorted(df_raw["likely_adopter"].unique()))

df = df_raw[
    df_raw["business_type"].isin(biz_filter)
    & df_raw["city_tier"].isin(tier_filter)
    & df_raw["annual_revenue_band"].isin(rev_filter)
    & df_raw["likely_adopter"].isin(adopt_filter)
].copy()

st.sidebar.markdown(
    f'<div style="background:rgba(44,122,75,0.10); padding:.7rem; '
    f'border-radius:8px; font-size:.85rem; color:{ECO_DEEP};">'
    f'<b>{len(df):,}</b> of {len(df_raw):,} businesses match your filters.'
    f'</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.download_button(
    "⬇️ Download filtered data",
    df.to_csv(index=False), "ecoverge_filtered.csv", "text/csv",
)

st.sidebar.markdown(
    f'<div style="font-size:.75rem; color:#5C6A65; margin-top:1rem;">'
    f'Synthetic survey data, n = 2,100 Indian B2B businesses.<br>'
    f'Built for an academic Business Data Analytics project.'
    f'</div>',
    unsafe_allow_html=True,
)


# ===========================================================================
# 1. HOME & OBJECTIVES
# ===========================================================================
if page == "🏠 Home & Objectives":
    hero(
        "From plastic to plant-based, by the numbers.",
        "Predicting which Indian B2B customers will adopt biodegradable packaging — "
        "and what it takes to win them.",
    )

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, f"{len(df_raw):,}", "Businesses surveyed")
    kpi_card(c2, f"{(df_raw['likely_adopter']=='Yes').mean()*100:.0f}%",
             "Likely to adopt", alt=True)
    kpi_card(c3, f"₹{df_raw['monthly_packaging_spend'].mean()/1000:.0f}k",
             "Avg monthly packaging spend")
    kpi_card(c4, f"₹{df_raw['expected_biodegradable_budget'].sum()/1e7:.1f} Cr",
             "Total addressable budget", warn=True)

    st.markdown("## The business problem")
    section_intro(
        "Indian businesses still default to plastic packaging because biodegradable "
        "alternatives feel expensive, unreliable, and untrusted. ECOVERGE's job is to "
        "turn that perception around — with the right products, the right pricing, and "
        "the right customers first. This dashboard turns 2,100 B2B survey responses into "
        "a concrete go-to-market plan."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 Project objectives")
        st.markdown(
            "1. Understand the Indian B2B market for biodegradable packaging.\n"
            "2. Identify segments most likely to adopt sustainable alternatives.\n"
            "3. Predict adoption likelihood via classification ML models.\n"
            "4. Estimate monthly packaging spend / biodegradable budget via regression.\n"
            "5. Segment businesses into actionable customer personas with clustering.\n"
            "6. Find product-bundle opportunities via association-rule mining.\n"
            "7. Forecast demand, revenue, and adoption growth.\n"
            "8. Recommend pricing, bundling, targeting, and sales strategies.\n"
            "9. Build a lead-scoring engine for new ECOVERGE prospects.\n"
            "10. Support sustainable, data-driven business growth at ≥ 20% per year."
        )
    with col_b:
        st.markdown("### 🌱 Who ECOVERGE serves")
        st.markdown(
            "ECOVERGE is a B2B biodegradable-packaging company supplying:\n\n"
            "- **Restaurants, cloud kitchens, cafés** — takeaway containers, bowls, cups, carry bags\n"
            "- **Hotels & caterers** — meal trays, premium branded packaging, food containers\n"
            "- **Retail stores** — carry bags, branded retail packaging\n"
            "- **E-commerce sellers & FMCG** — paper mailers, corrugated boxes, protective packaging\n"
        )
        st.markdown("### 📦 The product range")
        st.markdown(
            "Food containers · Plates · Bowls · Cups · Carry bags · Paper mailers · "
            "Corrugated/shipping boxes · Meal trays · Protective packaging · "
            "Premium branded retail packaging."
        )

    st.markdown("## The data-driven decision flow")
    flow = (
        "**Survey data** → **Cleaning** → **Feature engineering** → "
        "**Exploratory analysis** → **ML models** (classification / regression / "
        "clustering / rules) → **Business insights** → **Prescriptive recommendations** "
        "→ **Streamlit deployment**"
    )
    st.markdown(
        f'<div style="background:{ECO_SAND}; padding:1.2rem 1.4rem; border-radius:12px;'
        f' border:1px solid #E5E0CF; font-size:1.02rem; line-height:1.6;">{flow}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## What this dashboard answers")
    qa = [
        ("Which businesses are most likely to switch to biodegradable packaging?",
         "Classification + lead-scoring sections"),
        ("How much will they actually spend per month?",
         "Regression (budget prediction)"),
        ("What product bundles make sense for which type of buyer?",
         "Association rules + Recommender"),
        ("How fast can ECOVERGE realistically grow?",
         "Forecasting (next 12 months)"),
        ("Where should the sales team focus first?",
         "Clustering personas + Business recommendations"),
        ("Who can be turned into a referral champion?",
         "Referral network analysis"),
    ]
    for q, where in qa:
        st.markdown(f"- **{q}**  ·  *{where}*")


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================
elif page == "🧬 Feature Engineering":
    hero("Feature engineering",
         "10+ engineered scores translate raw survey answers into business signals.",
         tag="STEP 2 OF THE PIPELINE")

    section_intro(
        "Raw survey columns like <i>‘sustainability_importance = 4’</i> aren't very "
        "useful on their own. We combine them into <b>scores</b> that summarise how "
        "ready, how valuable, and how risky each business is — these scores then power "
        "every prediction and segmentation downstream."
    )

    features = [
        ("Adoption Readiness Score",
         "Sustainability mindset + supplier pain + customer-facing visibility − switching barriers + log(spend). "
         "Tells us how primed a business is to switch.",
         "adoption_readiness_score"),
        ("Sustainability Mindset Score",
         "Average of importance + plastic awareness + customer demand + ESG + regulatory pressure. "
         "Captures the depth of sustainability thinking.",
         "sustainability_mindset_score"),
        ("Supplier Pain Point Score",
         "Count of current supplier problems (cost, quality, delivery, eco options, MOQ, lead time). "
         "Higher pain = more open to switching.",
         "supplier_pain_score"),
        ("Switching Barrier Score",
         "Sum of barrier ratings (price, quality concern, current relationship, fear of customer rejection). "
         "The friction ECOVERGE must overcome.",
         "switching_barrier_score"),
        ("Trust Builder Score",
         "Count of confidence-builders the business says would convince them "
         "(samples, certification, pilot order, warranty, etc.).",
         "trust_builder_score"),
        ("Commercial Value Score",
         "log(monthly spend) × packaging-demand × customer visibility. Predicts revenue upside.",
         "commercial_value_score"),
        ("Packaging Demand Score",
         "Number of product categories the business is interested in. Wider interest = bigger basket.",
         "packaging_demand_score"),
        ("Price Sensitivity Score",
         "Price barrier × 1.3 − willingness-to-pay premium. Tells us whom to give discounts to.",
         "price_sensitivity_score"),
        ("Operational Risk Score",
         "How worried the business is about reliability and quality failures.",
         "operational_risk_score"),
        ("Customer Visibility Score",
         "Customer-facing + takeaway/delivery/retail/ecommerce + premium branding interest. "
         "High visibility = packaging is part of brand experience.",
         "customer_visibility_score"),
        ("Lead Priority Score",
         "Composite 0–100 score: adoption latent × spend × sustainability. The final lead-ranking signal.",
         "lead_priority_score"),
        ("Customer Value Segment",
         "Categorical: Low / Medium / High Value / Strategic Account. Used in targeting.",
         "customer_value_segment"),
        ("Adoption Risk Category",
         "Categorical: Cold / Warm / Hot / Champion Lead. Used by sales prioritisation.",
         "adoption_risk_category"),
        ("Product Fit Score",
         "Packaging demand × performance requirements alignment. Helps the recommender.",
         "product_fit_score"),
    ]

    for name, desc, col in features:
        st.markdown(f"#### {name}")
        st.markdown(f"<div style='color:#4A554F; line-height:1.55;'>{desc}</div>",
                    unsafe_allow_html=True)
        if col in df_raw.columns and df_raw[col].dtype.kind in "biufc":
            fig = px.histogram(df_raw, x=col, nbins=30,
                               color_discrete_sequence=[ECO_LEAF],
                               template=PLOTLY_TEMPLATE)
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title=name, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        elif col in df_raw.columns:
            counts = df_raw[col].value_counts().reset_index()
            counts.columns = [name, "count"]
            fig = px.bar(counts, x=name, y="count",
                         color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE)
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("")


# ===========================================================================
# 3. DESCRIPTIVE ANALYTICS
# ===========================================================================
elif page == "📊 Descriptive Analytics":
    hero("Descriptive analytics",
         "What the Indian B2B packaging market looks like, right now.",
         tag="WHAT HAPPENED")

    # Top KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, f"{len(df):,}", "Filtered respondents")
    kpi_card(c2, f"{(df['likely_adopter']=='Yes').mean()*100:.0f}%",
             "Likely-adopter rate", alt=True)
    kpi_card(c3, f"₹{df['monthly_packaging_spend'].mean()/1000:.0f}k", "Avg monthly spend")
    kpi_card(c4, f"₹{df['monthly_packaging_spend'].median()/1000:.0f}k",
             "Median monthly spend")
    kpi_card(c5, f"{df['monthly_order_volume'].mean()/1000:.1f}k",
             "Avg monthly order volume", warn=True)

    tab1, tab2, tab3 = st.tabs([
        "🏢 Business & market", "📦 Current packaging", "🎯 Product & sustainability"
    ])

    # ---- TAB 1: business & market
    with tab1:
        a, b = st.columns(2)
        with a:
            d = df["business_type"].value_counts().reset_index()
            d.columns = ["Business type", "Count"]
            fig = px.bar(d, x="Business type", y="Count",
                         color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                         title="Business type distribution")
            fig.update_layout(height=340, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            d = df["city_tier"].value_counts().reindex(["Tier 1", "Tier 2", "Tier 3"]).reset_index()
            d.columns = ["City tier", "Count"]
            fig = px.pie(d, names="City tier", values="Count", hole=0.45,
                         color_discrete_sequence=[ECO_LEAF, ECO_KRAFT, ECO_CLAY],
                         template=PLOTLY_TEMPLATE, title="City tier mix")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

        biz_note(
            "Restaurants and cloud kitchens dominate the sample; Tier 1 cities account "
            "for nearly half of respondents.",
            "Both groups are high-frequency packaging consumers — they define ECOVERGE's volume game.",
            "Anchor go-to-market on Tier 1 metros first (Delhi, Mumbai, Bengaluru), then expand to "
            "Tier 2 hubs in months 4-9."
        )

        a, b = st.columns(2)
        with a:
            d = df["annual_revenue_band"].value_counts().reset_index()
            d.columns = ["Revenue band", "Count"]
            fig = px.bar(d, x="Revenue band", y="Count",
                         color_discrete_sequence=[ECO_KRAFT], template=PLOTLY_TEMPLATE,
                         title="Annual revenue band")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            d = df["region"].value_counts().reset_index()
            d.columns = ["Region", "Count"]
            fig = px.bar(d, x="Region", y="Count",
                         color_discrete_sequence=[ECO_MOSS], template=PLOTLY_TEMPLATE,
                         title="Regional mix")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 2: current packaging
    with tab2:
        a, b = st.columns(2)
        with a:
            d = df["current_packaging_material"].value_counts().reset_index()
            d.columns = ["Material", "Count"]
            fig = px.bar(d, x="Material", y="Count",
                         color_discrete_sequence=[ECO_CLAY], template=PLOTLY_TEMPLATE,
                         title="What businesses currently use")
            fig.update_layout(height=340, xaxis_tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            d = df["order_frequency"].value_counts().reindex(
                ["Daily", "Weekly", "Bi-weekly", "Monthly", "Quarterly"]).reset_index()
            d.columns = ["Order frequency", "Count"]
            fig = px.bar(d, x="Order frequency", y="Count",
                         color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                         title="How often they order")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

        biz_note(
            "Plastic and mixed plastic-paper still dominate; weekly ordering is the modal cadence.",
            "These are exactly the customers ECOVERGE needs to convert — they buy often and they buy a lot.",
            "Design the supply contract around <b>weekly auto-replenishment</b> to fit existing operations."
        )

        fig = px.histogram(df, x="monthly_packaging_spend", nbins=50,
                           color_discrete_sequence=[ECO_DEEP], template=PLOTLY_TEMPLATE,
                           title="Distribution of monthly packaging spend (INR)")
        fig.update_layout(height=320, xaxis_title="Monthly spend (INR)",
                          yaxis_title="Number of businesses")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="business_type", y="monthly_packaging_spend",
                     color="business_type", color_discrete_sequence=ECO_SEQ,
                     template=PLOTLY_TEMPLATE, title="Spend by business type")
        fig.update_layout(height=380, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: product + sustainability
    with tab3:
        prefs = ["pref_food_containers", "pref_bowls", "pref_cups", "pref_carry_bags",
                 "pref_paper_mailers", "pref_corrugated_boxes", "pref_meal_trays",
                 "pref_protective_packaging", "pref_premium_branded"]
        nice = ["Food Containers", "Bowls", "Cups", "Carry Bags", "Paper Mailers",
                "Corrugated Boxes", "Meal Trays", "Protective Packaging", "Premium Branded"]
        rates = [(df[c] == "Yes").mean() * 100 for c in prefs]
        d = pd.DataFrame({"Product": nice, "Interest %": rates}).sort_values("Interest %", ascending=True)
        fig = px.bar(d, x="Interest %", y="Product", orientation="h",
                     color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                     title="Product interest (% of businesses saying Yes)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        biz_note(
            "Carry bags, food containers, bowls and meal trays top the demand list.",
            "These are ECOVERGE's volume products — high-frequency consumables with proven appetite.",
            "Launch with the top-5 product categories; treat premium-branded and paper-mailer SKUs as "
            "specialist add-ons for hotels and e-commerce buyers."
        )

        a, b = st.columns(2)
        with a:
            d = df["sustainability_importance"].value_counts().sort_index().reset_index()
            d.columns = ["Sustainability importance (1-5)", "Count"]
            fig = px.bar(d, x="Sustainability importance (1-5)", y="Count",
                         color_discrete_sequence=[ECO_MOSS], template=PLOTLY_TEMPLATE,
                         title="How much does sustainability matter?")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            d = df["adoption_intention"].value_counts().sort_index().reset_index()
            d.columns = ["Adoption intention (1-5)", "Count"]
            fig = px.bar(d, x="Adoption intention (1-5)", y="Count",
                         color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                         title="Adoption intention (next 6 months)")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Pain points + barriers + confidence builders
        st.markdown("### 🩹 Top supplier pain points")
        pain_cols = ["pain_high_cost", "pain_poor_quality", "pain_delayed_delivery",
                     "pain_inconsistent_quality", "pain_no_eco_options",
                     "pain_poor_customization", "pain_limited_reliability",
                     "pain_moq_issue", "pain_lead_time"]
        pain_nice = ["High cost", "Poor quality", "Delayed delivery",
                     "Inconsistent quality", "No eco options", "Poor customization",
                     "Limited reliability", "MOQ issues", "Long lead time"]
        pain_rates = [(df[c] == "Yes").mean() * 100 for c in pain_cols]
        d = pd.DataFrame({"Pain": pain_nice, "% experiencing": pain_rates}).sort_values("% experiencing", ascending=True)
        fig = px.bar(d, x="% experiencing", y="Pain", orientation="h",
                     color_discrete_sequence=[ECO_CLAY], template=PLOTLY_TEMPLATE,
                     title="Pain points with current packaging suppliers")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        biz_note(
            "High cost and lack of eco-friendly options are the two biggest current-supplier complaints.",
            "These are the openings ECOVERGE walks into — competitive pricing + a credible eco proposition flip the script.",
            "Lead with <b>price-matched pilot orders</b> and certification-backed eco claims in every sales conversation."
        )



# ===========================================================================
# 4. DIAGNOSTIC ANALYTICS — why are some businesses likely to adopt?
# ===========================================================================
elif page == "🔍 Diagnostic Analytics":
    hero("Diagnostic analytics",
         "Why do some businesses say yes to biodegradable, and others stall?",
         tag="WHY IT HAPPENED")

    section_intro(
        "We hold the descriptive picture next to the outcome — <i>likely-adopter Yes/No</i> — "
        "and look for the patterns that actually drive the decision."
    )

    # Adoption rate by business type
    by_biz = df.groupby("business_type")["likely_adopter"].apply(
        lambda s: (s == "Yes").mean() * 100).reset_index()
    by_biz.columns = ["Business type", "Adoption rate %"]
    by_biz = by_biz.sort_values("Adoption rate %", ascending=True)
    fig = px.bar(by_biz, x="Adoption rate %", y="Business type", orientation="h",
                 color="Adoption rate %", color_continuous_scale="Greens",
                 template=PLOTLY_TEMPLATE, title="Adoption rate by business type")
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    biz_note(
        "Hotels and e-commerce sellers show the highest adoption rates; retail stores lag.",
        "Hotels need ESG credentials and e-commerce needs paper mailers — both already "
        "have a sustainability story to tell their own customers.",
        "Make Hotels + E-commerce the <b>tip of the spear</b> for ECOVERGE's first six months."
    )

    # Adoption by city tier
    a, b = st.columns(2)
    with a:
        d = df.groupby("city_tier")["likely_adopter"].apply(
            lambda s: (s == "Yes").mean() * 100).reset_index()
        d.columns = ["City tier", "Adoption rate %"]
        fig = px.bar(d, x="City tier", y="Adoption rate %",
                     color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                     title="Adoption by city tier")
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        d = df.groupby("sustainability_importance")["likely_adopter"].apply(
            lambda s: (s == "Yes").mean() * 100).reset_index()
        d.columns = ["Sustainability importance", "Adoption rate %"]
        fig = px.line(d, x="Sustainability importance", y="Adoption rate %",
                      markers=True, color_discrete_sequence=[ECO_DEEP],
                      template=PLOTLY_TEMPLATE,
                      title="Sustainability importance → adoption")
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)

    biz_note(
        "Tier 1 adoption is roughly double Tier 3's. Importance score 5 buyers convert "
        "at ~8x the rate of importance score 1.",
        "Sustainability mindset is the single strongest qualitative predictor of adoption.",
        "Use sustainability importance as a <b>top-of-funnel qualifier</b> in every sales outreach."
    )

    # Spend & barriers
    a, b = st.columns(2)
    with a:
        fig = px.box(df, x="likely_adopter", y="monthly_packaging_spend",
                     color="likely_adopter",
                     color_discrete_sequence=[ECO_CLAY, ECO_LEAF],
                     template=PLOTLY_TEMPLATE,
                     title="Spend distribution by adoption intent")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.box(df, x="likely_adopter", y="barrier_higher_price",
                     color="likely_adopter",
                     color_discrete_sequence=[ECO_CLAY, ECO_LEAF],
                     template=PLOTLY_TEMPLATE,
                     title="Price-barrier rating by adoption intent")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    biz_note(
        "Likely adopters have higher monthly spend <b>and</b> lower price-sensitivity barriers.",
        "Bigger packaging budgets mean the cost-premium absorbs more easily.",
        "Target spend ≥ ₹40k/month accounts first; offer phased pricing for smaller buyers."
    )

    # Correlation heatmap of engineered scores
    st.markdown("### 🔥 Correlation heatmap — engineered scores vs adoption")
    df_cor = df.copy()
    df_cor["adopt"] = (df_cor["likely_adopter"] == "Yes").astype(int)
    cor_cols = ["adopt", "adoption_readiness_score", "sustainability_mindset_score",
                "supplier_pain_score", "switching_barrier_score", "trust_builder_score",
                "commercial_value_score", "packaging_demand_score",
                "price_sensitivity_score", "customer_visibility_score",
                "lead_priority_score", "monthly_packaging_spend"]
    cor = df_cor[cor_cols].corr().round(2)
    fig = px.imshow(cor, text_auto=True, color_continuous_scale="RdYlGn",
                    zmin=-1, zmax=1, aspect="auto", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    biz_note(
        "Adoption correlates strongly (>0.5) with adoption-readiness, sustainability mindset, "
        "and lead priority — and negatively with switching barriers.",
        "These four scores carry most of the predictive signal — they're the foundation of the ML models.",
        "Bake these four scores into ECOVERGE's CRM as the live lead-scoring fields."
    )

    # Cross-tab: business type × city tier × adoption
    st.markdown("### 🧩 Adoption by business type × city tier")
    ct = pd.crosstab([df["business_type"], df["city_tier"]],
                     df["likely_adopter"], normalize="index").round(3) * 100
    ct.columns = [f"{c} %" for c in ct.columns]
    st.dataframe(ct.reset_index(), use_container_width=True, hide_index=True)

    biz_note(
        "Hotels in Tier 1 cross 80% adoption; restaurants in Tier 3 sit near 15%.",
        "The opportunity is not 'restaurants vs hotels' — it's a 2-D grid where business "
        "type and geography multiply.",
        "Use this matrix to <b>literally rank ECOVERGE's first 1,000 sales calls</b>."
    )


# ===========================================================================
# 5. CLASSIFICATION
# ===========================================================================
elif page == "🤖 Classification Models":
    hero("Classification — who's going to adopt?",
         "Five models compared head-to-head on adoption prediction.",
         tag="PREDICTIVE ANALYTICS")

    section_intro(
        "Target = <b>likely_adopter</b> (Yes / No). We train Logistic Regression, "
        "Decision Tree, Random Forest, KNN, and Gradient Boosting on the engineered "
        "feature set (leakage columns removed), then compare on a held-out 25% test split."
    )

    with st.spinner("Training 5 classifiers on the full dataset…"):
        C = train_classifiers(df_raw)

    st.markdown("### 📊 Model comparison")
    res = C["results"].copy()
    st.dataframe(res, use_container_width=True, hide_index=True)

    best = res.iloc[0]["Model"]
    fmodel = C["fitted"][best]
    st.success(f"🏆 Best model: **{best}**  ·  F1 = {res.iloc[0]['F1']}  ·  AUC = {res.iloc[0]['AUC']}")

    a, b = st.columns(2)
    with a:
        cm = confusion_matrix(fmodel["yte"], fmodel["pred"])
        cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"],
                             columns=["Pred No", "Pred Yes"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens",
                        template=PLOTLY_TEMPLATE, aspect="auto",
                        title=f"Confusion matrix — {best}")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fpr, tpr, _ = roc_curve(fmodel["yte"], fmodel["proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 line=dict(color=ECO_LEAF, width=3),
                                 name=f"{best} (AUC={res.iloc[0]['AUC']})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="#888", dash="dash"),
                                 name="Random"))
        fig.update_layout(height=340, template=PLOTLY_TEMPLATE,
                          title="ROC curve",
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)

    biz_note(
        f"{best} delivers the best F1 ({res.iloc[0]['F1']}) and AUC ({res.iloc[0]['AUC']}).",
        f"That means ECOVERGE can confidently rank prospects by predicted adoption "
        f"probability — the model gets the order right ~{int(res.iloc[0]['AUC']*100)}% of the time.",
        "Deploy this model behind a lead-scoring API; refresh quarterly from new survey & sales data."
    )

    # Feature importance (use Random Forest for robust importances)
    if "Random Forest" in C["fitted"]:
        rf = C["fitted"]["Random Forest"]["model"]
        imp = pd.Series(rf.feature_importances_, index=C["cols"]).sort_values(ascending=False).head(15)
        imp_df = imp.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df = imp_df.sort_values("Importance", ascending=True)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Greens",
                     template=PLOTLY_TEMPLATE,
                     title="Top 15 most-important features (Random Forest)")
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        biz_note(
            "The top features are sustainability mindset, monthly spend, supplier pain, "
            "and customer visibility — exactly the engineered scores from the diagnostic view.",
            "The model is reading the right business signals, not arbitrary noise.",
            "Salespeople should ask three questions on every discovery call: <i>How important is "
            "sustainability to you? What's your monthly packaging spend? What's broken about your "
            "current supplier?</i> Those answers alone produce a strong adoption-probability estimate."
        )


# ===========================================================================
# 6. DECISION TREE
# ===========================================================================
elif page == "🌳 Decision Tree":
    hero("Decision tree — the rules behind adoption",
         "An explainable decision tree shows exactly which questions separate adopters from non-adopters.",
         tag="EXPLAINABLE AI")

    section_intro(
        "Black-box models predict well; a decision tree <b>explains</b>. This is the kind of "
        "model you can show a sales head or a co-founder and they'll instantly see the logic."
    )

    with st.spinner("Building decision tree…"):
        C = train_classifiers(df_raw)
    # Build a smaller, more interpretable tree just for this view
    X, y = C["X"], C["y"]
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, random_state=42).fit(X, y)

    # Visualise
    st.markdown("### 🌲 Tree visualisation")
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree, feature_names=C["cols"], class_names=["No", "Yes"],
              filled=True, rounded=True, fontsize=9, impurity=False, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### 📜 Decision rules (text)")
    rules_text = export_text(tree, feature_names=C["cols"], max_depth=4)
    st.text(rules_text[:4000])

    biz_note(
        "The very first split is on adoption-readiness / sustainability mindset; the second "
        "level brings in monthly packaging spend and supplier pain.",
        "These are the questions ECOVERGE should literally print on a sales scorecard.",
        "Convert the top-3 levels of this tree into a one-page <b>'Will they buy?'</b> qualification "
        "card for every new field sales rep."
    )



# ===========================================================================
# 7. CLUSTERING & PERSONAS
# ===========================================================================
elif page == "👥 Clustering & Personas":
    hero("Customer personas — what kinds of buyers exist?",
         "K-Means clustering finds natural groupings of businesses ECOVERGE can build playbooks around.",
         tag="SEGMENTATION")

    section_intro(
        "We cluster on engineered scores (adoption readiness, supplier pain, switching barriers, "
        "spend, product demand, customer visibility). Then we read each cluster's profile and name "
        "it like an actual sales persona."
    )

    k = st.slider("Number of clusters (K)", 3, 7, 4)

    with st.spinner("Clustering…"):
        C = run_clustering(df_raw, k=k)

    a, b = st.columns(2)
    with a:
        fig = px.line(x=C["k_range"], y=C["inertias"], markers=True,
                      color_discrete_sequence=[ECO_DEEP], template=PLOTLY_TEMPLATE,
                      title="Elbow method (within-cluster sum of squares)",
                      labels={"x": "K", "y": "Inertia"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(x=C["k_range"], y=C["sils"], markers=True,
                      color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                      title="Silhouette score",
                      labels={"x": "K", "y": "Silhouette"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Attach labels to df
    seg = df_raw.copy()
    seg["cluster"] = C["labels"]

    biz_note(
        f"At K = {k}, silhouette = {C['sils'][k-2]:.2f} — modest but typical for survey data "
        f"with overlapping segments. The elbow chart shows diminishing returns past K = 4.",
        "There's no single 'correct' K — pick the value that gives clusters with distinct "
        "business profiles, not just statistical separation.",
        "K = 4 is the operational sweet spot — enough variety to build different playbooks, "
        "few enough to actually staff and execute."
    )

    # Persona profiles
    st.markdown("### 🎭 Persona profiles")
    persona_names = {}
    for cid in sorted(seg["cluster"].unique()):
        sub = seg[seg["cluster"] == cid]
        # Auto-derive a persona name from the dominant traits
        top_biz = sub["business_type"].mode().iloc[0]
        sus = sub["sustainability_mindset_score"].mean()
        spend = sub["monthly_packaging_spend"].mean()
        readiness = sub["adoption_readiness_score"].mean()
        barrier = sub["switching_barrier_score"].mean()

        if readiness > 90 and sus > 3.6:
            name = f"🌟 Sustainability Champions"
        elif spend > 60000:
            name = f"💼 High-Spend Strategic Buyers"
        elif barrier > 26 and sus < 3.3:
            name = f"❄️ Price-Sensitive Holdouts"
        else:
            name = f"📈 Pragmatic Growth Buyers"
        persona_names[cid] = name

        st.markdown(
            f'<div class="persona-card">'
            f'<h4>Cluster {cid} · {name}</h4>'
            f'<span class="persona-tag">{len(sub)} businesses</span>'
            f'<span class="persona-tag">Most common type: {top_biz}</span>'
            f'<span class="persona-tag">Avg spend: ₹{spend/1000:.0f}k/mo</span>'
            f'<span class="persona-tag">Readiness: {readiness:.0f}</span>'
            f'<span class="persona-tag">Sustainability: {sus:.1f}/5</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Recommended strategy per persona
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Profile**")
            st.markdown(
                f"- Sustainability mindset: **{sus:.1f}/5**\n"
                f"- Avg monthly spend: **₹{spend:,.0f}**\n"
                f"- Adoption readiness: **{readiness:.0f}**\n"
                f"- Switching barrier: **{barrier:.0f}**\n"
                f"- Adoption rate: **{(sub['likely_adopter']=='Yes').mean()*100:.0f}%**"
            )
        with col2:
            st.markdown("**Recommended strategy**")
            if "Champions" in name:
                st.markdown(
                    "- **Bundle:** Premium Eco Suite (full range + branded retail)\n"
                    "- **Pricing:** Tier-1 list price, ESG case-study sweetener\n"
                    "- **Pitch:** *'You're already leading on sustainability — let's make your packaging match.'*\n"
                    "- **Priority:** HIGH"
                )
            elif "Strategic" in name:
                st.markdown(
                    "- **Bundle:** Volume + Reliability Bundle (food containers, bowls, mailers)\n"
                    "- **Pricing:** Bulk discount tier; quarterly contract\n"
                    "- **Pitch:** *'Reliable supply at scale, with a sustainability story for your investors.'*\n"
                    "- **Priority:** HIGH"
                )
            elif "Holdouts" in name:
                st.markdown(
                    "- **Bundle:** Trial Starter Pack (small MOQ, 2-3 SKUs)\n"
                    "- **Pricing:** Price-matched pilot, no premium for first 90 days\n"
                    "- **Pitch:** *'Try at the same price as plastic; only switch if it works.'*\n"
                    "- **Priority:** LOW until proven"
                )
            else:
                st.markdown(
                    "- **Bundle:** Core Bundle (carry bags, bowls, food containers)\n"
                    "- **Pricing:** Mid-range, with bulk-discount ladder\n"
                    "- **Pitch:** *'Eco packaging that performs — sample first, decide later.'*\n"
                    "- **Priority:** MEDIUM"
                )
        st.markdown("")

    # Spend × readiness scatter coloured by cluster
    st.markdown("### 🗺️ Persona map — spend vs adoption readiness")
    plot_df = seg.copy()
    plot_df["persona"] = plot_df["cluster"].map(persona_names)
    fig = px.scatter(plot_df, x="adoption_readiness_score",
                     y="monthly_packaging_spend", color="persona",
                     color_discrete_sequence=ECO_SEQ, template=PLOTLY_TEMPLATE,
                     log_y=True, opacity=0.62, height=460,
                     hover_data=["business_type", "city_tier"])
    fig.update_layout(legend_title="Persona")
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 8. ASSOCIATION RULES
# ===========================================================================
elif page == "🔗 Association Rules":
    hero("Association rules — which products go together?",
         "Apriori mines product-interest patterns. Bundles drop out of the data.",
         tag="MARKET-BASKET ANALYSIS")

    section_intro(
        "If businesses interested in <i>paper mailers</i> are also frequently interested in "
        "<i>corrugated boxes</i>, that's a natural bundle. Apriori finds these patterns "
        "automatically — we just have to read them and turn them into SKUs."
    )

    if not HAS_MLXTEND:
        st.warning("mlxtend isn't available in this environment. Add `mlxtend` to requirements.txt to enable this section.")
    else:
        st.markdown("**Quick legend**")
        st.markdown(
            "- **Support** — how common the combination is (e.g. 0.20 = 20% of businesses)\n"
            "- **Confidence** — given antecedent, probability of consequent\n"
            "- **Lift** — strength of association vs random chance (> 1 means real)"
        )

        c1, c2 = st.columns(2)
        sup = c1.slider("Minimum support", 0.05, 0.30, 0.12, 0.01)
        lift_min = c2.slider("Minimum lift", 1.0, 2.0, 1.05, 0.05)

        with st.spinner("Mining association rules…"):
            rules = mine_rules(df_raw, min_support=sup, min_lift=lift_min)

        if rules is None or rules.empty:
            st.warning("No rules found at these thresholds — try lowering support or lift.")
            st.stop()

        disp = rules.copy()
        disp["Antecedents"] = disp["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        disp["Consequents"] = disp["consequents"].apply(lambda s: ", ".join(sorted(s)))
        disp = disp[["Antecedents", "Consequents", "support", "confidence", "lift"]].round(3)
        disp.columns = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]

        st.markdown("### 🏆 Top rules")
        st.dataframe(disp.head(15), use_container_width=True, hide_index=True)

        plot_df = disp.head(40).rename(
            columns={"Support": "support", "Confidence": "confidence", "Lift": "lift"})
        fig = px.scatter(plot_df, x="support", y="confidence", size="lift", color="lift",
                         color_continuous_scale="Greens", template=PLOTLY_TEMPLATE,
                         hover_data={"Antecedents": True, "Consequents": True},
                         title="Rules: Support vs Confidence (bubble = Lift)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        top = disp.iloc[0]
        biz_note(
            f"Businesses preferring <b>{top['Antecedents']}</b> are strongly likely to also "
            f"prefer <b>{top['Consequents']}</b> (lift {top['Lift']}).",
            "These are organic product affinities — sell them as one SKU, not as separate line items.",
            "Package the top-lift pairs into named bundles — e.g. <b>'Takeaway Essentials'</b> "
            "(food containers + bowls + carry bags), <b>'Eco Shipping'</b> (paper mailers + "
            "corrugated boxes + protective packaging). Bundles convert higher than à la carte."
        )

        # Suggested named bundles
        st.markdown("### 📦 Suggested ECOVERGE bundles")
        st.markdown(
            "Based on the top association rules, three concrete bundles emerge:\n\n"
            "1. **🍱 Takeaway Essentials** — food containers + bowls + carry bags + cups. "
            "Target: restaurants, cloud kitchens, cafés.\n"
            "2. **📦 Eco Shipping** — paper mailers + corrugated boxes + protective packaging. "
            "Target: e-commerce sellers, FMCG, retail.\n"
            "3. **🏨 Premium Hospitality** — premium branded + bowls + meal trays + food containers. "
            "Target: hotels, premium caterers."
        )



# ===========================================================================
# 9. REGRESSION — predicting monthly biodegradable budget
# ===========================================================================
elif page == "💰 Regression (Budget Prediction)":
    hero("Regression — how much will they spend?",
         "Predicting each business's expected monthly biodegradable-packaging budget.",
         tag="PRICING & SIZING")

    section_intro(
        "We predict <b>expected_biodegradable_budget</b> (INR per month) using 4 regression "
        "models. The output drives pricing tiers, account sizing, and territory planning."
    )

    with st.spinner("Training 4 regression models…"):
        R = train_regressors(df_raw)

    st.markdown("### 📊 Regression model comparison")
    st.dataframe(R["results"], use_container_width=True, hide_index=True)
    best = R["results"].iloc[0]["Model"]
    st.success(f"🏆 Best model: **{best}**  ·  R² = {R['results'].iloc[0]['R²']}  ·  "
               f"MAE = ₹{R['results'].iloc[0]['MAE']:,}")

    # Actual vs predicted scatter
    reg = R["fitted"][best]
    pred = np.expm1(reg.predict(R["Xte"]))
    actual = np.expm1(R["yte"])
    fig = px.scatter(x=actual, y=pred, opacity=0.55,
                     color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                     labels={"x": "Actual budget (INR)", "y": "Predicted budget (INR)"},
                     title=f"Actual vs predicted — {best}")
    mx = max(actual.max(), pred.max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(color=ECO_DEEP, dash="dash"),
                             showlegend=False))
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    biz_note(
        f"{best} predicts monthly biodegradable budgets within ±₹{R['results'].iloc[0]['MAE']:,} "
        f"on average (R² = {R['results'].iloc[0]['R²']}).",
        "Account managers can quote ECOVERGE deals at the right ballpark from day one — "
        "no more guessing whether a hotel will spend ₹20k or ₹200k.",
        "Wire this into the CRM as an <b>auto-calculated 'expected deal size'</b> field."
    )

    # Feature importance for the best (or RF) regressor
    if hasattr(reg, "feature_importances_"):
        imp = pd.Series(reg.feature_importances_, index=R["cols"]).sort_values(ascending=False).head(12)
        imp_df = imp.reset_index(); imp_df.columns = ["Feature", "Importance"]
        imp_df = imp_df.sort_values("Importance", ascending=True)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Greens",
                     template=PLOTLY_TEMPLATE, title="Top features driving budget predictions")
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Pricing strategy recommendation
    st.markdown("### 💸 Pricing strategy implications")
    p25 = df_raw["expected_biodegradable_budget"].quantile(0.25)
    p50 = df_raw["expected_biodegradable_budget"].quantile(0.50)
    p75 = df_raw["expected_biodegradable_budget"].quantile(0.75)
    p90 = df_raw["expected_biodegradable_budget"].quantile(0.90)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, f"₹{p25/1000:.0f}k", "Entry tier (25%ile)")
    kpi_card(c2, f"₹{p50/1000:.0f}k", "Mid tier (median)", alt=True)
    kpi_card(c3, f"₹{p75/1000:.0f}k", "Premium tier (75%ile)")
    kpi_card(c4, f"₹{p90/1000:.0f}k", "Strategic tier (90%ile)", warn=True)

    biz_note(
        "The market splits naturally into four price-band tiers based on monthly budget.",
        "ECOVERGE doesn't need one price — it needs four, each matched to a tier's spend reality.",
        "<b>Entry:</b> small MOQ trial pricing for Micro/Small. <b>Mid:</b> standard bundle pricing. "
        "<b>Premium:</b> custom-branded packs. <b>Strategic:</b> annual contracts with dedicated account support."
    )


# ===========================================================================
# 10. FORECASTING
# ===========================================================================
elif page == "📈 Forecasting":
    hero("Forecasting — what does growth look like?",
         "Three forecasting methods project ECOVERGE sales 12 months ahead.",
         tag="DEMAND PLANNING")

    section_intro(
        "We simulate 24 months of ECOVERGE monthly unit sales (in '000s) with realistic "
        "seasonality, then forecast the next 12 months using three classic methods. The "
        "purpose is operational: how much inventory, how many people, how much working capital."
    )

    sales = simulate_sales()
    train = sales.copy()
    horizon = 12

    # 1) Moving average forecast
    ma = train["sales_k_units"].rolling(3).mean().iloc[-1]
    ma_forecast = [ma] * horizon

    # 2) Linear trend
    x = np.arange(len(train))
    coef = np.polyfit(x, train["sales_k_units"], 1)
    lin = np.polyval(coef, np.arange(len(train), len(train) + horizon))

    # 3) Exponential smoothing (simple)
    alpha = 0.4
    level = train["sales_k_units"].iloc[0]
    for v in train["sales_k_units"]:
        level = alpha * v + (1 - alpha) * level
    es_forecast = [level * (1 + 0.025) ** (i + 1) for i in range(horizon)]

    future_dates = pd.date_range(start=train["month"].iloc[-1] + pd.DateOffset(months=1),
                                 periods=horizon, freq="MS")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["month"], y=train["sales_k_units"], mode="lines+markers",
                             name="Historical", line=dict(color=ECO_DEEP, width=3)))
    fig.add_trace(go.Scatter(x=future_dates, y=ma_forecast, mode="lines+markers",
                             name="Moving Average", line=dict(color=ECO_KRAFT, dash="dot")))
    fig.add_trace(go.Scatter(x=future_dates, y=lin, mode="lines+markers",
                             name="Linear Trend", line=dict(color=ECO_LEAF, dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=es_forecast, mode="lines+markers",
                             name="Exp. Smoothing + 2.5% growth", line=dict(color=ECO_CLAY)))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=440,
                      title="ECOVERGE monthly sales — historical + 12-month forecast",
                      xaxis_title="Month", yaxis_title="Sales ('000 units)")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast KPIs
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, f"{train['sales_k_units'].iloc[-1]:.0f}k", "Last month sales")
    kpi_card(c2, f"{lin[-1]:.0f}k", "12-mo linear trend forecast", alt=True)
    kpi_card(c3, f"{(lin[-1]/train['sales_k_units'].iloc[0]-1)*100:.0f}%",
             "Projected growth from month 1", warn=True)

    biz_note(
        "Linear trend and exponential smoothing both project comfortably above the 20% "
        "annual growth target ECOVERGE has set as its baseline.",
        "Growth is achievable — but only if operations and supply chain scale alongside.",
        "Use the highest forecast as the <b>inventory ceiling</b> and the lowest as the "
        "<b>working-capital floor</b>. Plan around the spread, not the point estimate."
    )

    st.markdown("### 🗓️ Operational implications")
    st.markdown(
        "- **Inventory** — pre-stock 1.5× the lowest-month historical sales going into festival quarters.\n"
        "- **Production** — line up a secondary contract manufacturer before month 6.\n"
        "- **Sales hiring** — close two BDR hires by month 4 to handle pipeline expansion.\n"
        "- **Marketing** — front-load campaigns into months with the seasonal lift (Oct–Dec)."
    )


# ===========================================================================
# 11. RECOMMENDER SYSTEM
# ===========================================================================
elif page == "🎁 Recommender System":
    hero("Product recommender",
         "Profile a business → get a tailored ECOVERGE bundle, pricing, and sales pitch.",
         tag="PERSONALISATION")

    section_intro(
        "Pick a business profile and the rule-based recommender suggests the right product "
        "bundle, the right discount lever, and the right sales pitch. This is the kind of "
        "logic you'd embed in ECOVERGE's CRM."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        biz = st.selectbox("Business type", sorted(df_raw["business_type"].unique()))
    with c2:
        tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    with c3:
        size = st.selectbox("Business size", ["Micro", "Small", "Medium", "Large"])

    c1, c2, c3 = st.columns(3)
    with c1:
        sustain = st.slider("Sustainability importance", 1, 5, 4)
    with c2:
        spend = st.number_input("Monthly packaging spend (INR)", 5000, 500000, 35000, 5000)
    with c3:
        price_concern = st.slider("Price-sensitivity (1=low, 5=high)", 1, 5, 3)

    # Rule-based recommender
    bundles = []
    if biz in ("Restaurant", "Cloud Kitchen", "Café", "Caterer"):
        bundle = "🍱 Takeaway Essentials Bundle"
        items = "Food containers · Bowls · Cups · Carry bags · Meal trays"
    elif biz in ("E-commerce Seller", "FMCG / Food Processing"):
        bundle = "📦 Eco Shipping Bundle"
        items = "Paper mailers · Corrugated boxes · Protective packaging · Carry bags"
    elif biz == "Hotel":
        bundle = "🏨 Premium Hospitality Bundle"
        items = "Premium branded bowls · Meal trays · Food containers · Custom-print carry bags"
    elif biz == "Retail Store":
        bundle = "🛍️ Retail Storefront Bundle"
        items = "Carry bags · Premium branded packaging · Paper mailers"
    else:
        bundle = "🌿 Core Eco Bundle"
        items = "Food containers · Bowls · Carry bags · Paper mailers"

    # Pricing lever
    if price_concern >= 4:
        discount = "Trial pricing — 0% premium for 90 days, then 8% premium"
    elif spend >= 80000:
        discount = "Volume contract — 12% bulk discount on annual commitment"
    elif sustain >= 4:
        discount = "Standard premium pricing — sustainability story sells the gap"
    else:
        discount = "Mid-tier pricing with quarterly bulk discount ladder"

    # Pitch
    if sustain >= 4 and price_concern <= 3:
        pitch = ("\"Your sustainability commitment deserves packaging that lives up to it. "
                 "ECOVERGE certified-compostable products with ESG-grade documentation — "
                 "ready to deploy this quarter.\"")
    elif price_concern >= 4:
        pitch = ("\"We'll match your current packaging cost for the first 90 days. "
                 "Test our quality on your own customers; only switch fully if it works.\"")
    elif spend >= 80000:
        pitch = ("\"For your scale, we'll set up a dedicated account, guaranteed delivery, "
                 "and a custom co-branded retail line — at 12% off list with an annual contract.\"")
    else:
        pitch = ("\"Sample-led trial, performance-guaranteed delivery, and a bundle that "
                 "covers 80% of your packaging needs in one PO. Let's start with samples this week.\"")

    # Lead priority
    if sustain >= 4 and spend >= 50000:
        priority_label = "HOT LEAD"
        priority_class = "lead-hot"
    elif sustain >= 3 and spend >= 25000:
        priority_label = "WARM LEAD"
        priority_class = "lead-warm"
    else:
        priority_label = "COLD LEAD"
        priority_class = "lead-cold"

    st.markdown("### 🎯 ECOVERGE recommendation")
    st.markdown(
        f'<div class="persona-card">'
        f'<h4>{bundle}</h4>'
        f'<p style="color:#4A554F;">{items}</p>'
        f'<p><b>Recommended discount:</b> {discount}</p>'
        f'<p><b>Sales pitch:</b> <i>{pitch}</i></p>'
        f'<p><b>Lead priority:</b> <span class="{priority_class}">{priority_label}</span></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    biz_note(
        "A single profile produces a complete go-to-market recommendation — bundle, pricing, "
        "pitch, and priority.",
        "ECOVERGE's sales team doesn't need to think on its feet — the playbook is pre-decided.",
        "Surface this same logic inside the CRM, so any rep can pull the right move within seconds."
    )


# ===========================================================================
# 12. TEXT MINING & SENTIMENT
# ===========================================================================
elif page == "💬 Text Mining & Sentiment":
    hero("Voice of the customer",
         "What businesses actually say about packaging — and how it shapes ECOVERGE's response.",
         tag="UNSTRUCTURED DATA")

    section_intro(
        "Survey feedback comments are run through VADER sentiment analysis. The themes "
        "tell ECOVERGE what to fix, what to highlight, and what to stop talking about."
    )

    if not HAS_VADER:
        st.warning("VADER sentiment library isn't available. Add `vaderSentiment` to requirements.txt.")
    else:
        analyzer = SentimentIntensityAnalyzer()
        texts = df["feedback_text"].dropna().astype(str).tolist()
        compounds = [analyzer.polarity_scores(t)["compound"] for t in texts]
        sentiment = pd.Series(np.where(
            np.array(compounds) > 0.05, "Positive",
            np.where(np.array(compounds) < -0.05, "Negative", "Neutral")
        ))

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, f"{(sentiment == 'Positive').mean()*100:.0f}%", "Positive feedback")
        kpi_card(c2, f"{(sentiment == 'Neutral').mean()*100:.0f}%", "Neutral", alt=True)
        kpi_card(c3, f"{(sentiment == 'Negative').mean()*100:.0f}%", "Negative", warn=True)

        a, b = st.columns(2)
        with a:
            d = sentiment.value_counts().reindex(["Positive", "Neutral", "Negative"]).reset_index()
            d.columns = ["Sentiment", "Count"]
            fig = px.bar(d, x="Sentiment", y="Count",
                         color="Sentiment",
                         color_discrete_map={"Positive": ECO_LEAF, "Neutral": ECO_KRAFT,
                                             "Negative": ECO_CLAY},
                         template=PLOTLY_TEMPLATE, title="Sentiment distribution")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.histogram(x=compounds, nbins=40, color_discrete_sequence=[ECO_MOSS],
                               template=PLOTLY_TEMPLATE,
                               title="Compound sentiment score distribution")
            fig.update_layout(height=340, xaxis_title="VADER compound score",
                              yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # Word cloud
        if HAS_WORDCLOUD:
            st.markdown("### ☁️ Word cloud of feedback")
            all_text = " ".join(texts)
            wc = WordCloud(width=1200, height=400, background_color=ECO_SAND,
                           colormap="Greens", max_words=80,
                           stopwords={"and", "the", "to", "for", "we", "is", "be",
                                      "of", "in", "a", "with", "on", "our", "this",
                                      "are", "as", "from", "but", "have", "that",
                                      "it", "an", "or", "if", "us", "their"}
                           ).generate(all_text)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Install `wordcloud` to display the word cloud here.")

        # Themes
        st.markdown("### 🧭 Themes from positive feedback")
        st.markdown(
            "- Openness to switching <i>if pricing is competitive</i>\n"
            "- Eco-friendly packaging seen as future-proofing the brand\n"
            "- Strong appetite for pilot orders and samples\n"
            "- Reliability and consistent quality framed as deciding factors"
        )

        st.markdown("### 🩹 Themes from negative feedback")
        st.markdown(
            "- Cost-premium concerns and margin pressure\n"
            "- Past bad experiences with leaky paper containers\n"
            "- Greenwashing fatigue — claims need lab-test proof\n"
            "- MOQ and lead-time complaints with eco-suppliers"
        )

        biz_note(
            "Customers want the eco story — but only if it doesn't cost more and doesn't fail.",
            "ECOVERGE's central messaging needs to neutralise the cost and reliability objections "
            "before it sells the sustainability dream.",
            "Build all sales collateral around three claims: <b>(1) same total cost as plastic at scale, "
            "(2) lab-tested durability, (3) certified compostability</b>. Lead with proof, not with adjectives."
        )



# ===========================================================================
# 13. REFERRAL NETWORK
# ===========================================================================
elif page == "🌐 Referral Network":
    hero("Referral network",
         "Which businesses are most likely to bring more businesses to ECOVERGE?",
         tag="NETWORK ANALYTICS")

    section_intro(
        "We build a synthetic B2B referral graph from the top 80 lead-priority businesses. "
        "NetworkX identifies the most influential hubs — the customers worth turning into "
        "anchor accounts."
    )

    G = build_referral_graph(df_raw, max_nodes=80)

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, f"{G.number_of_nodes()}", "Businesses in network")
    kpi_card(c2, f"{G.number_of_edges()}", "Referral edges", alt=True)
    avg_deg = sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
    kpi_card(c3, f"{avg_deg:.1f}", "Avg degree per business", warn=True)

    # Layout + plot
    pos = nx.spring_layout(G, seed=42, k=0.6)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [10 + G.degree(n) * 3 for n in G.nodes()]
    node_color = [G.nodes[n]["score"] for n in G.nodes()]
    node_text = [f"{n}<br>{G.nodes[n]['biz']}<br>Score: {G.nodes[n]['score']:.0f}<br>"
                 f"Referrals: {G.nodes[n]['referrals']}" for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(140,140,140,0.45)", width=0.6),
                             hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                             marker=dict(size=node_size, color=node_color,
                                         colorscale="Greens", showscale=True,
                                         colorbar=dict(title="Lead score"),
                                         line=dict(color="white", width=1)),
                             text=node_text, hoverinfo="text", showlegend=False))
    fig.update_layout(height=560, template=PLOTLY_TEMPLATE,
                      title="ECOVERGE referral network (node size = influence)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

    # Most influential nodes
    centrality = nx.degree_centrality(G)
    inf = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    inf_rows = []
    for nid, c in inf:
        n = G.nodes[nid]
        inf_rows.append({
            "Business ID": nid, "Business type": n["biz"],
            "Lead score": round(n["score"], 1), "Referrals out": n["referrals"],
            "Centrality": round(c, 3),
        })
    st.markdown("### 🌟 Top 10 most influential businesses")
    st.dataframe(pd.DataFrame(inf_rows), use_container_width=True, hide_index=True)

    # Referrals by business type
    type_refs = df_raw.groupby("business_type")["referral_count"].mean().reset_index().sort_values(
        "referral_count", ascending=True)
    fig = px.bar(type_refs, x="referral_count", y="business_type", orientation="h",
                 color_discrete_sequence=[ECO_LEAF], template=PLOTLY_TEMPLATE,
                 title="Average referrals by business type")
    fig.update_layout(height=380, xaxis_title="Avg referrals out",
                      yaxis_title="Business type")
    st.plotly_chart(fig, use_container_width=True)

    biz_note(
        "Hotels and e-commerce sellers don't just buy — they influence others to buy.",
        "These segments compound ECOVERGE's growth: every win unlocks multiple downstream conversations.",
        "Build a formal <b>'Eco Anchor'</b> referral programme — invite the top 50 hotel and "
        "e-commerce customers, offer them rebates per referred account, and feature them as case studies."
    )


# ===========================================================================
# 14. ETHICS & SUSTAINABILITY
# ===========================================================================
elif page == "⚖️ Ethics & Sustainability":
    hero("Ethics, sustainability & AI governance",
         "How ECOVERGE intends to be credible, transparent, and genuinely green.",
         tag="RESPONSIBILITY")

    section_intro(
        "ECOVERGE sits at the intersection of two things people are right to be skeptical "
        "about — sustainability claims and AI-driven decisions. This section lays out how "
        "the business plans to earn trust on both fronts."
    )

    t1, t2, t3 = st.tabs(["🔐 Data ethics & privacy", "🤖 Responsible AI",
                          "🌱 Sustainability & ESG"])

    with t1:
        st.markdown("### Data ethics & privacy commitments")
        st.markdown(
            "- **Informed consent** — every survey response is collected with a clear, "
            "plain-English purpose statement.\n"
            "- **Anonymisation** — business names and contact details are stripped from the "
            "analytical dataset; only de-identified business profiles are modelled.\n"
            "- **Secure storage** — survey data is held in an encrypted database, with role-"
            "based access for the analytics team only.\n"
            "- **Right to withdraw** — any respondent can request deletion of their data at "
            "any time.\n"
            "- **Purpose limitation** — data collected for the ECOVERGE GTM project is not "
            "repurposed for marketing email lists without separate explicit consent."
        )

    with t2:
        st.markdown("### Responsible AI in ECOVERGE")
        st.markdown(
            "- **Explainable lead scoring** — the production model is paired with a decision "
            "tree and feature-importance view so any sales rep can see *why* a lead was ranked "
            "hot or cold.\n"
            "- **Bias checks** — adoption-rate parity is monitored across city tiers and "
            "business sizes; if any segment is systematically under-served by the model, the "
            "model is retrained.\n"
            "- **Human-in-the-loop** — predictions are a guide for sales prioritisation, not "
            "an automatic gate. A rep can always override.\n"
            "- **Model refresh cadence** — every quarter, on a fresh wave of survey + closed-"
            "deal data, so the model stays current as the market evolves.\n"
            "- **No personal data in features** — the model uses business attributes only, "
            "never individual contact details."
        )

    with t3:
        st.markdown("### Sustainability & ESG commitments")
        st.markdown(
            "- **Certified compostable / biodegradable** — every product line ships with "
            "third-party lab certifications (CPCB, BIS, or equivalent).\n"
            "- **No greenwashing** — claims are backed with test reports, not adjectives. "
            "Marketing copy is reviewed against an internal evidence standard.\n"
            "- **Sustainable sourcing** — raw materials traceable to farms or recycled-paper "
            "mills; supplier audits annually.\n"
            "- **End-of-life** — partner with municipal composting programs in Tier 1 cities "
            "so packaging doesn't end up in landfill by default.\n"
            "- **ESG reporting** — annual impact report measuring tonnes of plastic "
            "displaced, CO₂ equivalent saved, and circular-economy outcomes."
        )

        # Quick impact illustration
        plastic_displaced_kg = df_raw["expected_biodegradable_budget"].sum() / 80  # rough INR-to-kg proxy
        co2_saved_t = plastic_displaced_kg * 0.0025
        st.markdown("### 📊 Indicative environmental impact")
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, f"~{plastic_displaced_kg/1000:.1f} t/mo",
                 "Plastic potentially displaced")
        kpi_card(c2, f"~{co2_saved_t:.0f} t/mo", "CO₂-equivalent saved", alt=True)
        kpi_card(c3, "100%",
                 "Products certified compostable", warn=True)

        biz_note(
            "Even at conservative conversion, the modelled customer base could displace "
            "multiple tonnes of plastic per month.",
            "ESG impact is real, measurable, and reportable — not just a marketing claim.",
            "Publish a public sustainability dashboard from year 1. Transparency is itself a "
            "competitive moat against greenwashing competitors."
        )


# ===========================================================================
# 15. BUSINESS RECOMMENDATIONS
# ===========================================================================
elif page == "🧭 Business Recommendations":
    hero("So what should ECOVERGE actually do?",
         "A founder-style summary of the entire dashboard.",
         tag="PRESCRIPTIVE")

    section_intro(
        "Strip away the models and the charts; here's what the data is telling ECOVERGE's "
        "founders to do, in order."
    )

    recs = [
        ("🎯 Lead with hotels, e-commerce sellers, cafés, and cloud kitchens",
         "Adoption rates in these segments range from 50% to 70% — roughly 3× the rate "
         "for retail stores. These are the wedge segments for the first 6 months."),
        ("🏙️ Anchor on Tier 1 metros first, then expand to Tier 2",
         "Tier 1 adoption is roughly double Tier 3's. Concentrate sales feet on the ground "
         "in Delhi, Mumbai, Bengaluru, Pune, Hyderabad before expanding outward."),
        ("📦 Launch with five SKUs, not fifty",
         "Carry bags, food containers, bowls, paper mailers, and corrugated boxes cover "
         "80%+ of demand across the top business types. Premium-branded SKUs come in Q2 "
         "as a hospitality-segment up-sell."),
        ("💸 Sell at four price tiers — entry, mid, premium, strategic",
         "The expected-budget distribution splits cleanly into quartiles. One blanket price "
         "would lose either margin (at the top) or volume (at the bottom)."),
        ("🧪 Make samples and pilot orders the default opening move",
         "78% of businesses say samples would build their confidence; 72% would do a pilot. "
         "Open every account with a paid-trial pack, not a quote."),
        ("📜 Lead with certification and lab proof, not adjectives",
         "Greenwashing concern is high; trust in biodegradable claims is moderate. Marketing "
         "should foreground compostability certificates and food-safety lab reports."),
        ("🎁 Sell bundles, not items",
         "Association rules surface three obvious bundles: Takeaway Essentials, Eco Shipping, "
         "Premium Hospitality. Bundles convert higher than à la carte."),
        ("🌟 Build a referral programme around hotels and e-commerce sellers",
         "These two segments score highest on network centrality. Anchor accounts in these "
         "verticals unlock additional accounts at low CAC."),
        ("📊 Wire ML lead scoring into the CRM from day one",
         "The classifier explains ~80%+ of adoption variance from a small set of business "
         "attributes. Every inbound lead should be scored before the first sales call."),
        ("📈 Plan for 20%+ annual growth — but back it with operations",
         "Forecast models all clear the 20% target. Hire BDRs by month 4, line up a backup "
         "manufacturer by month 6, and pre-stock 1.5× the lowest-month inventory before "
         "festival quarters."),
    ]

    for title, body in recs:
        st.markdown(f"#### {title}")
        st.markdown(f"<div style='color:#4A554F; line-height:1.6; margin-bottom:1.3rem;'>{body}</div>",
                    unsafe_allow_html=True)

    # Founder-style closing
    st.markdown(
        f'<div style="background:{ECO_DEEP}; color:#F8F5EC; padding: 1.6rem 1.8rem; '
        f'border-radius: 14px; margin-top: 1.5rem;">'
        f'<h3 style="color:#F8F5EC; margin-top:0;">The bet</h3>'
        f'<p style="font-size:1.05rem; line-height:1.7;">'
        f'ECOVERGE wins by being the first biodegradable packaging supplier in India that is '
        f'<i>obsessive about reliability and proof</i>, not just about adjectives. Lab-tested '
        f'durability, certified compostability, price parity at pilot scale, and a sales motion '
        f'that ranks every prospect before the first call. Do that, and the data says ECOVERGE '
        f'can capture a meaningful slice of the ~₹2,000 Cr addressable B2B packaging market '
        f'inside three years.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# 16. NEW CUSTOMER / LEAD PREDICTION
# ===========================================================================
elif page == "🆕 New Customer / Lead Prediction":
    hero("Lead scoring — score a new business in seconds",
         "Either fill in a single business profile, or upload a CSV of new leads.",
         tag="DEPLOY THE MODEL")

    with st.spinner("Loading scoring models…"):
        C = train_classifiers(df_raw)
        R = train_regressors(df_raw)
    clf_name = C["results"].iloc[0]["Model"]
    clf = C["fitted"][clf_name]["model"]
    clf_scale = C["fitted"][clf_name]["scale"]
    reg_name = R["results"].iloc[0]["Model"]
    reg = R["fitted"][reg_name]

    def predict_one(profile_overrides):
        """Build a single-row frame from dataset medians/modes + overrides, predict."""
        row = {}
        for col in df_raw.columns:
            if col in LEAK_COLS:
                continue
            if df_raw[col].dtype.kind in "biufc":
                row[col] = df_raw[col].median()
            else:
                row[col] = df_raw[col].mode().iloc[0]
        row.update(profile_overrides)
        one = pd.DataFrame([row])
        # classification path
        Xc = pd.get_dummies(one.drop(columns=[c for c in LEAK_COLS if c in one.columns]),
                            drop_first=True).reindex(columns=C["cols"], fill_value=0)
        Xc_in = C["scaler"].transform(Xc) if clf_scale else Xc
        proba = float(clf.predict_proba(Xc_in)[:, 1][0])
        # regression path (exclude monthly_packaging_spend + premium so we don't leak)
        Xr_extra = ["expected_biodegradable_budget", "monthly_packaging_spend",
                    "willingness_to_pay_premium_pct"]
        Xr = pd.get_dummies(one.drop(columns=[c for c in (LEAK_COLS + Xr_extra)
                                              if c in one.columns]),
                            drop_first=True).reindex(columns=R["cols"], fill_value=0)
        budget_log = float(reg.predict(Xr)[0])
        return proba, clamp_budget(np.expm1(budget_log))

    tab1, tab2 = st.tabs(["✍️ Single business", "📤 Upload CSV (batch)"])

    # ---- Single business form
    with tab1:
        section_intro("Score a single prospective ECOVERGE customer by entering their profile below.")

        c1, c2, c3 = st.columns(3)
        with c1:
            biz_type = st.selectbox("Business type", sorted(df_raw["business_type"].unique()))
            city_tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
            biz_size = st.selectbox("Business size", ["Micro", "Small", "Medium", "Large"])
        with c2:
            n_emp = st.number_input("Employees", 1, 800, 25)
            spend = st.number_input("Current monthly packaging spend (INR)",
                                    1000, 500000, 30000, 5000)
            order_freq = st.selectbox("Order frequency",
                                      ["Daily", "Weekly", "Bi-weekly", "Monthly", "Quarterly"])
        with c3:
            sustain = st.slider("Sustainability importance (1-5)", 1, 5, 4)
            price_barrier = st.slider("Price barrier (1=low, 5=high)", 1, 5, 3)
            cust_facing = st.selectbox("Customer-facing packaging?", ["Yes", "No"])

        if st.button("🎯 Score this lead", type="primary"):
            ov = {
                "business_type": biz_type, "city_tier": city_tier,
                "business_size": biz_size, "n_employees": n_emp,
                "monthly_packaging_spend": spend, "order_frequency": order_freq,
                "sustainability_importance": sustain,
                "barrier_higher_price": price_barrier,
                "customer_facing_packaging": cust_facing,
            }
            proba, budget = predict_one(ov)

            if proba >= 0.65 and budget >= 40000:
                segment, klass = "Champion Lead", "lead-hot"
            elif proba >= 0.5:
                segment, klass = "Hot Lead", "lead-hot"
            elif proba >= 0.3:
                segment, klass = "Warm Lead", "lead-warm"
            else:
                segment, klass = "Cold Lead", "lead-cold"

            st.markdown("### 🎯 ECOVERGE lead scoring result")
            c1, c2, c3 = st.columns(3)
            kpi_card(c1, f"{proba*100:.0f}%", "Adoption likelihood")
            kpi_card(c2, f"₹{budget:,.0f}", "Predicted monthly budget", alt=True)
            kpi_card(c3, segment, "Lead segment", warn=True)

            st.markdown(
                f'<div class="biz-note">'
                f'<span class="label">Sales next step</span>'
                f'<span class="{klass}">{segment}</span> &nbsp; — '
                f'{"Schedule a discovery call this week; lead with samples and certification." if proba>=0.5 else "Park in nurture pipeline; revisit after a quarter of email engagement."}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---- Batch CSV upload
    with tab2:
        section_intro(
            "Upload a CSV with the same column names as the ECOVERGE training data. "
            "Each row will be scored for adoption likelihood and predicted monthly budget. "
            "Missing columns are filled with safe defaults, so partial CSVs still work."
        )
        st.download_button(
            "⬇️ Download a sample CSV template",
            df_raw.drop(columns=LEAK_COLS, errors="ignore").head(5).to_csv(index=False),
            "ecoverge_sample_template.csv", "text/csv",
        )

        up = st.file_uploader("Upload new-lead CSV", type=["csv"])
        if up is not None:
            try:
                new = pd.read_csv(up)
                Xc = pd.get_dummies(
                    new.drop(columns=[c for c in LEAK_COLS if c in new.columns],
                             errors="ignore"),
                    drop_first=True
                ).reindex(columns=C["cols"], fill_value=0)
                Xc_in = C["scaler"].transform(Xc) if clf_scale else Xc
                proba = clf.predict_proba(Xc_in)[:, 1]

                Xr_extra = ["expected_biodegradable_budget", "monthly_packaging_spend",
                            "willingness_to_pay_premium_pct"]
                Xr = pd.get_dummies(
                    new.drop(columns=[c for c in (LEAK_COLS + Xr_extra)
                                       if c in new.columns], errors="ignore"),
                    drop_first=True
                ).reindex(columns=R["cols"], fill_value=0)
                budget = np.clip(np.expm1(reg.predict(Xr)), BUDGET_LO, BUDGET_HI)

                out = new.copy()
                out["Adoption_Likelihood_%"] = (proba * 100).round(1)
                out["Predicted_Monthly_Budget_INR"] = budget.round(0).astype(int)
                out["Lead_Segment"] = np.where(
                    proba >= 0.65, "Champion Lead",
                    np.where(proba >= 0.5, "Hot Lead",
                             np.where(proba >= 0.3, "Warm Lead", "Cold Lead"))
                )

                st.success(f"Scored {len(out)} leads.")
                st.dataframe(out.head(25), use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download scored leads",
                    out.to_csv(index=False),
                    "ecoverge_scored_leads.csv", "text/csv",
                )

                # Summary
                a, b, c = st.columns(3)
                kpi_card(a, f"{(out['Lead_Segment'].isin(['Hot Lead','Champion Lead'])).sum()}",
                         "Hot + Champion leads")
                kpi_card(b, f"₹{int(out['Predicted_Monthly_Budget_INR'].sum()):,}",
                         "Total predicted budget / month", alt=True)
                kpi_card(c, f"{out['Adoption_Likelihood_%'].mean():.0f}%",
                         "Avg adoption likelihood", warn=True)

                biz_note(
                    "Every uploaded lead now has an adoption likelihood, an expected budget, "
                    "and a priority segment.",
                    "ECOVERGE can prioritise sales outreach scientifically — no more 'gut feel' "
                    "prospecting.",
                    "Feed exported sales-team prospect lists through this scorer weekly to keep "
                    "the pipeline ranked by data, not by the latest meeting."
                )
            except Exception as e:
                st.error(
                    f"Could not score the file. Make sure column names match the template. "
                    f"({e})"
                )


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align:center; color:#5C6A65; font-size:.8rem; padding:1rem;">'
    f'<b style="color:{ECO_DEEP};">ECOVERGE</b> · Biodegradable packaging analytics · '
    f'Synthetic survey data · Built for a Business Data Analytics assessment.</div>',
    unsafe_allow_html=True,
)
