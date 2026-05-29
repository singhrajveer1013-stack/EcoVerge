# 🌿 ECOVERGE — Biodegradable Packaging Analytics Dashboard

A complete **Business Data Analytics** dashboard for ECOVERGE, a proposed B2B
biodegradable-packaging company in India. The dashboard turns ~2,100 simulated
B2B survey responses into a full go-to-market plan — from descriptive KPIs and
diagnostic deep-dives through predictive ML, customer personas, association
rules, forecasting, recommender logic, sentiment analysis, network analysis,
ethics, and lead scoring.

Built for an academic Business Data Analytics assessment project.

---

## 🎯 What's inside

16 analytics sections, each with business interpretations after every chart:

1. **Home & Objectives** — the ECOVERGE business problem and what the dashboard answers
2. **Feature Engineering** — 14 engineered scores translated into business signals
3. **Descriptive Analytics** — market structure, current packaging, product interest, sustainability mindset
4. **Diagnostic Analytics** — why some businesses adopt and others don't (correlation, segment splits)
5. **Classification Models** — 5 models (LogReg / Decision Tree / Random Forest / KNN / Gradient Boosting) predicting `likely_adopter`
6. **Decision Tree** — interpretable rule extraction with full tree visualisation
7. **Clustering & Personas** — K-Means segmentation with auto-named personas and per-segment sales strategies
8. **Association Rules** — Apriori bundle mining on product preferences
9. **Regression (Budget Prediction)** — predicting monthly biodegradable-packaging budget in INR
10. **Forecasting** — 12-month projection using moving average + linear trend + exponential smoothing
11. **Recommender System** — profile → bundle + pricing + sales pitch + lead priority
12. **Text Mining & Sentiment** — VADER sentiment on simulated B2B feedback
13. **Referral Network** — NetworkX graph of B2B referral influence
14. **Ethics & Sustainability** — data ethics, responsible AI, ESG commitments
15. **Business Recommendations** — founder-style summary of the entire dashboard
16. **New Customer / Lead Prediction** — single-business form + batch CSV scoring

---

## 📁 Files

| File | Purpose |
|---|---|
| `app.py` | The complete Streamlit dashboard (≈ 2,100 lines, 16 sections) |
| `generate_data.py` | Reproducible synthetic B2B respondent generator (seed = 42) |
| `ecoverge_data.csv` | Cleaned synthetic dataset, 2,100 rows × 100 columns |
| `requirements.txt` | Python dependencies (loose-pinned for cloud deploy) |
| `README.md` | This file |
| `.gitignore` | Standard Python ignores |

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

To rebuild the synthetic dataset from scratch (optional — the CSV ships with the project):

```bash
python generate_data.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. Click **"New app"**, pick the repo, set the main file to `app.py`.
4. Click **Deploy**.

The dependencies in `requirements.txt` use **lower-bound** version constraints
(e.g. `numpy>=2.1`) rather than exact pins. This lets the installer pick the
newest releases, which ship pre-built wheels for current Python versions
(including Python 3.14 used by Streamlit Community Cloud) — so the build
resolves cleanly with no compilation step and no dependency errors. The
cleaned dataset (`ecoverge_data.csv`) ships with the project, so the app runs
immediately without a generation step.

---

## 🧪 Synthetic data disclosure

`ecoverge_data.csv` is **synthetic** — it was procedurally generated to mimic
realistic Indian B2B packaging survey responses across 9 business types,
3 city tiers, 5 revenue bands, and a coherent adoption-intention signal.
Distributions, correlations, and segment-level dynamics are calibrated to be
business-sensible (e.g. hotels and e-commerce sellers show higher adoption,
sustainability importance drives intention monotonically, monthly spend
scales with business size). All names, IDs, and feedback comments are
simulated. No real customer or business data is used.

---

## 🏷️ ECOVERGE positioning

ECOVERGE is positioned as a **B2B sustainable packaging supplier** serving:
restaurants, cloud kitchens, cafés, caterers, hotels, retail stores,
e-commerce sellers, and FMCG/food-processing businesses across India's
Tier 1/2/3 cities. Product range: food containers, plates, bowls, cups,
carry bags, paper mailers, corrugated/shipping boxes, meal trays,
protective packaging, and premium branded retail packaging.

---

## 📚 Tech stack

Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Matplotlib ·
NetworkX · mlxtend · WordCloud · vaderSentiment.

---

*Built as a Business Data Analytics assessment project — synthetic data, no
real customer information used.*
