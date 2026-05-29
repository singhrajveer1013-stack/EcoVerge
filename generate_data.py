"""
generate_data.py
================
Builds a synthetic B2B survey dataset for the ECOVERGE biodegradable packaging
project. Produces ~2,100 Indian business respondents across 9 business types,
3 city tiers, and revenue/size bands, with engineered scores and a coherent
adoption-intention target.

Seed = 42 for reproducibility. Output: ecoverge_data.csv (cleaned).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2100

# ---------------------------------------------------------------------------
# 1. CATEGORICAL DOMAINS — what kinds of businesses do we survey?
# ---------------------------------------------------------------------------

BUSINESS_TYPES = [
    "Restaurant", "Cloud Kitchen", "Café", "Caterer", "Hotel",
    "Retail Store", "E-commerce Seller", "FMCG / Food Processing", "Other",
]
# weighted so restaurants + cloud kitchens dominate (realistic for India)
BUSINESS_WEIGHTS = [0.21, 0.17, 0.10, 0.07, 0.08, 0.12, 0.13, 0.08, 0.04]

CITY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
CITY_WEIGHTS = [0.45, 0.36, 0.19]

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.24, 0.26, 0.14, 0.26, 0.10]

REVENUE_BANDS = ["< 50L", "50L - 2 Cr", "2 - 10 Cr", "10 - 50 Cr", "> 50 Cr"]
BUSINESS_SIZES = ["Micro", "Small", "Medium", "Large"]

CURRENT_MATERIALS = [
    "Plastic", "Mixed (Plastic + Paper)", "Paper / Cardboard",
    "Biodegradable (Partial)", "Aluminium / Foil",
]
SUPPLIER_TYPES = ["Local Wholesaler", "Direct Manufacturer", "Distributor",
                  "Online Marketplace", "Multiple Small Suppliers"]

ORDER_FREQUENCIES = ["Daily", "Weekly", "Bi-weekly", "Monthly", "Quarterly"]
DECISION_MAKERS = ["Owner", "Operations Manager", "Procurement Head",
                   "Chef / Kitchen Manager", "Founder / Director", "Committee"]


def yes_no(p_yes, n=N):
    return np.where(rng.random(n) < p_yes, "Yes", "No")


def likert(p_high, p_med, n=N, labels=("Low", "Medium", "High")):
    """3-level Likert based on probability of High and Medium."""
    u = rng.random(n)
    out = np.where(u < p_high, labels[2],
                   np.where(u < p_high + p_med, labels[1], labels[0]))
    return out


# ---------------------------------------------------------------------------
# 2. BUSINESS PROFILE
# ---------------------------------------------------------------------------

biz_type = rng.choice(BUSINESS_TYPES, size=N, p=BUSINESS_WEIGHTS)
city_tier = rng.choice(CITY_TIERS, size=N, p=CITY_WEIGHTS)
region = rng.choice(REGIONS, size=N, p=REGION_WEIGHTS)
years_op = np.clip(rng.gamma(shape=2.0, scale=4.0, size=N).round().astype(int), 1, 35)
n_employees = np.clip(rng.gamma(shape=1.4, scale=42, size=N).round().astype(int), 2, 800)

# revenue is correlated with employee count + city tier
emp_factor = np.log1p(n_employees) / np.log1p(600)
tier_boost = np.where(city_tier == "Tier 1", 0.18,
                      np.where(city_tier == "Tier 2", 0.05, -0.05))
rev_score = emp_factor + tier_boost + rng.normal(0, 0.12, N)
revenue_band = pd.cut(
    rev_score,
    bins=[-99, 0.30, 0.50, 0.68, 0.85, 99],
    labels=REVENUE_BANDS,
).astype(str)

business_size = np.where(
    n_employees < 10, "Micro",
    np.where(n_employees < 50, "Small",
             np.where(n_employees < 200, "Medium", "Large")))

# Customer-facing packaging requirement
cust_facing_p = {
    "Restaurant": 0.92, "Cloud Kitchen": 0.96, "Café": 0.88,
    "Caterer": 0.78, "Hotel": 0.84, "Retail Store": 0.62,
    "E-commerce Seller": 0.95, "FMCG / Food Processing": 0.70, "Other": 0.55,
}
cust_facing = np.array([
    "Yes" if rng.random() < cust_facing_p[b] else "No" for b in biz_type
])

# ---------------------------------------------------------------------------
# 3. DECISION-MAKING
# ---------------------------------------------------------------------------

# More employees → more stakeholders, longer approval
n_stakeholders = np.clip(
    1 + (n_employees / 60) + rng.normal(0, 0.7, N), 1, 8).round().astype(int)
decision_maker = rng.choice(DECISION_MAKERS, size=N,
                            p=[0.32, 0.22, 0.16, 0.12, 0.10, 0.08])
supplier_approval = likert(0.32, 0.45)  # how formal is the process
purchase_complexity = likert(0.28, 0.50)
days_to_approve = np.clip(
    np.round(3 + (n_stakeholders * 4) + rng.normal(0, 6, N)), 1, 90).astype(int)
owner_involvement = likert(0.45, 0.35,
                           labels=("Rarely", "Sometimes", "Always"))

# ---------------------------------------------------------------------------
# 4. CURRENT PACKAGING USAGE
# ---------------------------------------------------------------------------

# Material mix correlates with business type
def pick_material(b):
    if b in ("Restaurant", "Cloud Kitchen", "Caterer"):
        return rng.choice(CURRENT_MATERIALS, p=[0.50, 0.30, 0.10, 0.07, 0.03])
    if b == "Hotel":
        return rng.choice(CURRENT_MATERIALS, p=[0.30, 0.30, 0.20, 0.15, 0.05])
    if b == "Café":
        return rng.choice(CURRENT_MATERIALS, p=[0.42, 0.28, 0.18, 0.10, 0.02])
    if b == "E-commerce Seller":
        return rng.choice(CURRENT_MATERIALS, p=[0.32, 0.32, 0.25, 0.10, 0.01])
    if b == "Retail Store":
        return rng.choice(CURRENT_MATERIALS, p=[0.40, 0.30, 0.18, 0.10, 0.02])
    if b == "FMCG / Food Processing":
        return rng.choice(CURRENT_MATERIALS, p=[0.36, 0.30, 0.16, 0.10, 0.08])
    return rng.choice(CURRENT_MATERIALS, p=[0.40, 0.25, 0.20, 0.10, 0.05])


current_material = np.array([pick_material(b) for b in biz_type])

# Monthly packaging spend in INR — driven by business size, type, city tier
type_spend_base = {
    "Restaurant": 22000, "Cloud Kitchen": 35000, "Café": 14000,
    "Caterer": 28000, "Hotel": 55000, "Retail Store": 18000,
    "E-commerce Seller": 42000, "FMCG / Food Processing": 70000, "Other": 16000,
}
size_mult = {"Micro": 0.45, "Small": 1.0, "Medium": 2.2, "Large": 4.8}
tier_mult = {"Tier 1": 1.20, "Tier 2": 1.00, "Tier 3": 0.78}

monthly_spend = np.array([
    type_spend_base[b] * size_mult[s] * tier_mult[t]
    * float(rng.lognormal(mean=0.0, sigma=0.30))
    for b, s, t in zip(biz_type, business_size, city_tier)
]).round().astype(int)

order_freq = rng.choice(ORDER_FREQUENCIES, size=N,
                        p=[0.18, 0.42, 0.18, 0.18, 0.04])
# monthly order volume (units of packaging consumed per month)
freq_mult = {"Daily": 30, "Weekly": 4.3, "Bi-weekly": 2.2, "Monthly": 1, "Quarterly": 0.33}
order_volume = np.array([
    max(200, int(monthly_spend[i] / rng.uniform(7, 18) * freq_mult[order_freq[i]] / 30))
    for i in range(N)
])

current_supplier = rng.choice(SUPPLIER_TYPES, size=N,
                              p=[0.34, 0.22, 0.20, 0.16, 0.08])

uses_plastic = np.where(
    np.isin(current_material, ["Plastic", "Mixed (Plastic + Paper)"]), "Yes",
    np.where(rng.random(N) < 0.18, "Yes", "No"))

# Use cases — what does the business need packaging for?
def usecase_p(b):
    base = {"takeaway": 0.4, "dine_in": 0.4, "delivery": 0.4, "retail": 0.3,
            "ecommerce": 0.2, "gifting": 0.1, "storage": 0.4, "transport": 0.3}
    if b in ("Restaurant", "Cloud Kitchen", "Café"):
        base["takeaway"] = 0.92; base["delivery"] = 0.90; base["dine_in"] = 0.78
    if b == "Caterer":
        base["takeaway"] = 0.85; base["transport"] = 0.92; base["delivery"] = 0.7
    if b == "Hotel":
        base["dine_in"] = 0.90; base["retail"] = 0.55; base["gifting"] = 0.35
    if b == "E-commerce Seller":
        base["ecommerce"] = 0.96; base["transport"] = 0.85; base["gifting"] = 0.30
    if b == "Retail Store":
        base["retail"] = 0.92; base["takeaway"] = 0.50; base["gifting"] = 0.20
    if b == "FMCG / Food Processing":
        base["storage"] = 0.95; base["transport"] = 0.92; base["retail"] = 0.70
    return base


usecase_takeaway = np.array(["Yes" if rng.random() < usecase_p(b)["takeaway"] else "No" for b in biz_type])
usecase_dinein = np.array(["Yes" if rng.random() < usecase_p(b)["dine_in"] else "No" for b in biz_type])
usecase_delivery = np.array(["Yes" if rng.random() < usecase_p(b)["delivery"] else "No" for b in biz_type])
usecase_retail = np.array(["Yes" if rng.random() < usecase_p(b)["retail"] else "No" for b in biz_type])
usecase_ecommerce = np.array(["Yes" if rng.random() < usecase_p(b)["ecommerce"] else "No" for b in biz_type])
usecase_gifting = np.array(["Yes" if rng.random() < usecase_p(b)["gifting"] else "No" for b in biz_type])
usecase_storage = np.array(["Yes" if rng.random() < usecase_p(b)["storage"] else "No" for b in biz_type])
usecase_transport = np.array(["Yes" if rng.random() < usecase_p(b)["transport"] else "No" for b in biz_type])

# ---------------------------------------------------------------------------
# 5. PRODUCT INTEREST (the "basket" for association rules)
# ---------------------------------------------------------------------------

def pref_p(b):
    p = {"food_containers": 0.45, "bowls": 0.40, "cups": 0.35, "carry_bags": 0.50,
         "paper_mailers": 0.18, "corrugated_boxes": 0.30, "meal_trays": 0.35,
         "protective_packaging": 0.22, "premium_branded": 0.25}
    if b in ("Restaurant", "Cloud Kitchen"):
        p["food_containers"] = 0.82; p["bowls"] = 0.80; p["meal_trays"] = 0.72
        p["carry_bags"] = 0.74; p["cups"] = 0.55
    if b == "Café":
        p["cups"] = 0.88; p["carry_bags"] = 0.72; p["food_containers"] = 0.55
        p["premium_branded"] = 0.52
    if b == "Caterer":
        p["meal_trays"] = 0.85; p["bowls"] = 0.70; p["food_containers"] = 0.78
    if b == "Hotel":
        p["premium_branded"] = 0.70; p["bowls"] = 0.55; p["meal_trays"] = 0.50
        p["food_containers"] = 0.60
    if b == "E-commerce Seller":
        p["paper_mailers"] = 0.85; p["corrugated_boxes"] = 0.88
        p["protective_packaging"] = 0.78; p["premium_branded"] = 0.42
    if b == "Retail Store":
        p["carry_bags"] = 0.85; p["premium_branded"] = 0.45
        p["corrugated_boxes"] = 0.40
    if b == "FMCG / Food Processing":
        p["corrugated_boxes"] = 0.82; p["protective_packaging"] = 0.65
        p["food_containers"] = 0.50
    return p


pref_food_containers = np.array(["Yes" if rng.random() < pref_p(b)["food_containers"] else "No" for b in biz_type])
pref_bowls = np.array(["Yes" if rng.random() < pref_p(b)["bowls"] else "No" for b in biz_type])
pref_cups = np.array(["Yes" if rng.random() < pref_p(b)["cups"] else "No" for b in biz_type])
pref_carry_bags = np.array(["Yes" if rng.random() < pref_p(b)["carry_bags"] else "No" for b in biz_type])
pref_paper_mailers = np.array(["Yes" if rng.random() < pref_p(b)["paper_mailers"] else "No" for b in biz_type])
pref_corrugated_boxes = np.array(["Yes" if rng.random() < pref_p(b)["corrugated_boxes"] else "No" for b in biz_type])
pref_meal_trays = np.array(["Yes" if rng.random() < pref_p(b)["meal_trays"] else "No" for b in biz_type])
pref_protective_packaging = np.array(["Yes" if rng.random() < pref_p(b)["protective_packaging"] else "No" for b in biz_type])
pref_premium_branded = np.array(["Yes" if rng.random() < pref_p(b)["premium_branded"] else "No" for b in biz_type])

# ---------------------------------------------------------------------------
# 6. PERFORMANCE REQUIREMENTS (importance Likert 1-5)
# ---------------------------------------------------------------------------

def likert_5(mu, sigma=1.0, n=N):
    return np.clip(np.round(rng.normal(mu, sigma, n)), 1, 5).astype(int)


food_safe_req = likert_5(4.4, 0.7)
leak_proof_req = likert_5(4.1, 0.9)
heat_resistance_req = likert_5(3.6, 1.1)
strength_req = likert_5(4.0, 0.9)
design_branding_req = likert_5(3.2, 1.2)
compostability_req = likert_5(3.4, 1.3)
appearance_req = likert_5(3.6, 1.1)

# Hotels and Cafés care more about appearance/branding
mask = np.isin(biz_type, ["Hotel", "Café"])
design_branding_req = np.where(mask, np.clip(design_branding_req + 1, 1, 5), design_branding_req)
appearance_req = np.where(mask, np.clip(appearance_req + 1, 1, 5), appearance_req)
premium_factor = np.where(mask, 1, 0)

# ---------------------------------------------------------------------------
# 7. SUPPLIER PAIN POINTS (Yes/No - currently experiencing this pain)
# ---------------------------------------------------------------------------

pain_high_cost = yes_no(0.55)
pain_poor_quality = yes_no(0.42)
pain_delayed_delivery = yes_no(0.38)
pain_inconsistent_quality = yes_no(0.40)
pain_no_eco_options = yes_no(0.46)
pain_poor_customization = yes_no(0.32)
pain_limited_reliability = yes_no(0.35)
pain_moq_issue = yes_no(0.42)
pain_lead_time = yes_no(0.40)

# ---------------------------------------------------------------------------
# 8. SUSTAINABILITY & TRUST
# ---------------------------------------------------------------------------

# Sustainability importance influenced by business type and city tier
def sustain_mu(b, t):
    base = 3.0
    if b in ("Hotel", "Café", "E-commerce Seller"):
        base += 0.8
    if b in ("Restaurant", "Cloud Kitchen"):
        base += 0.3
    if t == "Tier 1":
        base += 0.4
    elif t == "Tier 3":
        base -= 0.3
    return base


sustainability_importance = np.array([
    int(np.clip(np.round(rng.normal(sustain_mu(b, t), 0.9)), 1, 5))
    for b, t in zip(biz_type, city_tier)
])
plastic_awareness = likert_5(3.8, 0.9)
customer_demand_eco = likert_5(3.3, 1.0)
regulatory_pressure_aware = likert_5(3.2, 1.1)
esg_image_importance = likert_5(3.4, 1.1)

# Adjust ESG up for hotels/e-commerce
esg_image_importance = np.where(
    np.isin(biz_type, ["Hotel", "E-commerce Seller"]),
    np.clip(esg_image_importance + 1, 1, 5), esg_image_importance)

cert_required = yes_no(0.62)
lab_test_required = yes_no(0.55)
trust_biodegradable_claims = likert_5(3.0, 1.0)
greenwashing_concern = likert_5(3.5, 1.1)

# ---------------------------------------------------------------------------
# 9. SWITCHING BARRIERS (1-5)
# ---------------------------------------------------------------------------

barrier_higher_price = likert_5(3.7, 1.0)
barrier_quality_concern = likert_5(3.6, 1.0)
barrier_current_relationship = likert_5(3.2, 1.1)
barrier_lack_trusted_suppliers = likert_5(3.3, 1.1)
barrier_supply_inconsistency = likert_5(3.5, 1.0)
barrier_lack_awareness = likert_5(2.8, 1.1)
barrier_customer_rejection_fear = likert_5(2.6, 1.1)
barrier_operational_risk = likert_5(3.1, 1.0)

# ---------------------------------------------------------------------------
# 10. CONFIDENCE BUILDERS (Yes/No - would this convince them?)
# ---------------------------------------------------------------------------

cb_samples = yes_no(0.78)
cb_pilot_order = yes_no(0.72)
cb_competitive_pricing = yes_no(0.85)
cb_bulk_discount = yes_no(0.74)
cb_quality_warranty = yes_no(0.80)
cb_certification_proof = yes_no(0.70)
cb_delivery_reliability = yes_no(0.78)
cb_custom_branding = yes_no(0.55)
cb_flexible_payment = yes_no(0.62)

# ---------------------------------------------------------------------------
# 11. LATENT ADOPTION SIGNAL → target variables
# ---------------------------------------------------------------------------

def norm01(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


# Composite adoption latent
adoption_latent = (
    0.20 * norm01(sustainability_importance)
    + 0.10 * norm01(customer_demand_eco)
    + 0.10 * norm01(esg_image_importance)
    + 0.10 * norm01(plastic_awareness)
    + 0.10 * norm01(np.log1p(monthly_spend))
    + 0.08 * (cust_facing == "Yes").astype(float)
    + 0.08 * (np.isin(biz_type, ["Hotel", "Café", "E-commerce Seller", "Cloud Kitchen"])).astype(float)
    + 0.06 * (city_tier == "Tier 1").astype(float)
    + 0.05 * ((pain_no_eco_options == "Yes").astype(float)
              + (pain_high_cost == "Yes").astype(float)) / 2
    - 0.12 * norm01(barrier_higher_price)
    - 0.10 * norm01(barrier_quality_concern)
    - 0.05 * norm01(barrier_current_relationship)
    + rng.normal(0, 0.09, N)
)

# Threshold at the 60th percentile so ~40% are "likely adopters"
threshold = np.quantile(adoption_latent, 0.60)
likely_adopter = np.where(adoption_latent >= threshold, "Yes", "No")

# 5-point adoption intention scale
adoption_intention = pd.cut(
    adoption_latent,
    bins=[-99, np.quantile(adoption_latent, 0.20),
          np.quantile(adoption_latent, 0.40),
          np.quantile(adoption_latent, 0.60),
          np.quantile(adoption_latent, 0.85), 99],
    labels=[1, 2, 3, 4, 5],
).astype(int)

# Expected biodegradable budget = function of current spend + sustainability + size,
# discounted by price barrier
exp_budget = (
    monthly_spend
    * (0.65 + 0.10 * norm01(sustainability_importance)
       + 0.08 * norm01(esg_image_importance)
       - 0.10 * norm01(barrier_higher_price))
    * np.exp(rng.normal(0, 0.18, N))
).round().astype(int)
exp_budget = np.clip(exp_budget, 1500, 800000)

# Willingness-to-pay premium over current packaging (in %)
wtp_premium = (
    5 + 12 * norm01(sustainability_importance)
    + 8 * norm01(customer_demand_eco)
    + 6 * norm01(esg_image_importance)
    - 10 * norm01(barrier_higher_price)
    + rng.normal(0, 4, N)
).round(1)
wtp_premium = np.clip(wtp_premium, -5, 35)

# Lead priority score (0-100) — composite, used as continuous lead score
lead_priority_score = (
    100 * norm01(adoption_latent) * 0.55
    + 100 * norm01(np.log1p(monthly_spend)) * 0.30
    + 100 * norm01(sustainability_importance) * 0.15
).round(1)

# ---------------------------------------------------------------------------
# 12. SIMULATED CUSTOMER FEEDBACK TEXT
# ---------------------------------------------------------------------------

pos_templates = [
    "Open to switching if pricing is competitive and quality is proven.",
    "We need eco-friendly packaging to meet customer expectations.",
    "Sustainability is a big priority for our brand image this year.",
    "Looking for reliable suppliers with consistent quality and good lead times.",
    "Happy to try a pilot order if samples are good and certifications check out.",
    "Eco-friendly is the future, we want to be early movers in our segment.",
    "Bulk pricing and dependable delivery will be the deciding factor for us.",
    "Our customers ask for sustainable packaging, so this is a serious option.",
]
neg_templates = [
    "Biodegradable options are too expensive for our margins right now.",
    "Quality of eco packaging we've tried before was inconsistent and disappointing.",
    "Our current supplier is reliable, switching feels risky and unnecessary.",
    "Greenwashing is everywhere, hard to trust biodegradable claims without proof.",
    "Lead times and minimum order quantities are usually a problem with eco suppliers.",
    "We tried paper containers but they leaked and customers complained.",
    "Cost pressure is too high to absorb a price premium for green packaging.",
    "Need strong evidence of food-safety and durability before we even consider it.",
]
neutral_templates = [
    "Interested but need more information on pricing and durability.",
    "We are evaluating options, no immediate decision planned this quarter.",
    "Need to see samples and understand the supply terms before committing.",
    "Open to discussion but cost-benefit needs to make sense for our scale.",
    "Considering this as part of our broader sustainability roadmap.",
]


def make_feedback(i):
    score = adoption_latent[i]
    if score > np.quantile(adoption_latent, 0.65):
        return rng.choice(pos_templates)
    if score < np.quantile(adoption_latent, 0.35):
        return rng.choice(neg_templates)
    return rng.choice(neutral_templates)


feedback_text = np.array([make_feedback(i) for i in range(N)])

# ---------------------------------------------------------------------------
# 13. REFERRAL DATA (for network analysis)
# ---------------------------------------------------------------------------

# Influential nodes get more referrals
influence = norm01(lead_priority_score) * 0.6 + (cust_facing == "Yes") * 0.2 + rng.random(N) * 0.2
referral_count = np.clip(np.round(influence * 8 + rng.normal(0, 1.2, N)), 0, 15).astype(int)

# ---------------------------------------------------------------------------
# 14. ASSEMBLE DATAFRAME
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    "respondent_id": [f"ECV{1000 + i}" for i in range(N)],

    # Business profile
    "business_type": biz_type,
    "city_tier": city_tier,
    "region": region,
    "years_of_operation": years_op,
    "n_employees": n_employees,
    "annual_revenue_band": revenue_band,
    "business_size": business_size,
    "customer_facing_packaging": cust_facing,

    # Decision-making
    "decision_maker": decision_maker,
    "n_decision_stakeholders": n_stakeholders,
    "supplier_approval_formality": supplier_approval,
    "purchase_decision_complexity": purchase_complexity,
    "days_to_approve_supplier": days_to_approve,
    "owner_involvement": owner_involvement,

    # Current packaging
    "current_packaging_material": current_material,
    "monthly_packaging_spend": monthly_spend,
    "order_frequency": order_freq,
    "monthly_order_volume": order_volume,
    "current_supplier_type": current_supplier,
    "uses_plastic_packaging": uses_plastic,
    "usecase_takeaway": usecase_takeaway,
    "usecase_dinein": usecase_dinein,
    "usecase_delivery": usecase_delivery,
    "usecase_retail": usecase_retail,
    "usecase_ecommerce": usecase_ecommerce,
    "usecase_gifting": usecase_gifting,
    "usecase_storage": usecase_storage,
    "usecase_transport": usecase_transport,

    # Product preferences
    "pref_food_containers": pref_food_containers,
    "pref_bowls": pref_bowls,
    "pref_cups": pref_cups,
    "pref_carry_bags": pref_carry_bags,
    "pref_paper_mailers": pref_paper_mailers,
    "pref_corrugated_boxes": pref_corrugated_boxes,
    "pref_meal_trays": pref_meal_trays,
    "pref_protective_packaging": pref_protective_packaging,
    "pref_premium_branded": pref_premium_branded,

    # Performance requirements
    "req_food_safe": food_safe_req,
    "req_leak_proof": leak_proof_req,
    "req_heat_resistance": heat_resistance_req,
    "req_strength_durability": strength_req,
    "req_design_branding": design_branding_req,
    "req_compostability": compostability_req,
    "req_appearance": appearance_req,

    # Pain points
    "pain_high_cost": pain_high_cost,
    "pain_poor_quality": pain_poor_quality,
    "pain_delayed_delivery": pain_delayed_delivery,
    "pain_inconsistent_quality": pain_inconsistent_quality,
    "pain_no_eco_options": pain_no_eco_options,
    "pain_poor_customization": pain_poor_customization,
    "pain_limited_reliability": pain_limited_reliability,
    "pain_moq_issue": pain_moq_issue,
    "pain_lead_time": pain_lead_time,

    # Sustainability & trust
    "sustainability_importance": sustainability_importance,
    "plastic_impact_awareness": plastic_awareness,
    "customer_demand_eco": customer_demand_eco,
    "regulatory_pressure_awareness": regulatory_pressure_aware,
    "esg_image_importance": esg_image_importance,
    "certification_required": cert_required,
    "lab_test_required": lab_test_required,
    "trust_biodegradable_claims": trust_biodegradable_claims,
    "greenwashing_concern": greenwashing_concern,

    # Switching barriers
    "barrier_higher_price": barrier_higher_price,
    "barrier_quality_concern": barrier_quality_concern,
    "barrier_current_relationship": barrier_current_relationship,
    "barrier_lack_trusted_suppliers": barrier_lack_trusted_suppliers,
    "barrier_supply_inconsistency": barrier_supply_inconsistency,
    "barrier_lack_awareness": barrier_lack_awareness,
    "barrier_customer_rejection_fear": barrier_customer_rejection_fear,
    "barrier_operational_risk": barrier_operational_risk,

    # Confidence builders
    "cb_samples": cb_samples,
    "cb_pilot_order": cb_pilot_order,
    "cb_competitive_pricing": cb_competitive_pricing,
    "cb_bulk_discount": cb_bulk_discount,
    "cb_quality_warranty": cb_quality_warranty,
    "cb_certification_proof": cb_certification_proof,
    "cb_delivery_reliability": cb_delivery_reliability,
    "cb_custom_branding": cb_custom_branding,
    "cb_flexible_payment": cb_flexible_payment,

    # Targets / outcomes
    "adoption_intention": adoption_intention,
    "likely_adopter": likely_adopter,
    "expected_biodegradable_budget": exp_budget,
    "willingness_to_pay_premium_pct": wtp_premium,
    "lead_priority_score": lead_priority_score,

    # Feedback + referrals
    "feedback_text": feedback_text,
    "referral_count": referral_count,
})

# ---------------------------------------------------------------------------
# 15. INJECT REALISTIC NOISE, THEN CLEAN
# ---------------------------------------------------------------------------

# Introduce a handful of missing values
for col, frac in [("annual_revenue_band", 0.012),
                  ("customer_demand_eco", 0.008),
                  ("days_to_approve_supplier", 0.010),
                  ("current_supplier_type", 0.006)]:
    miss_idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[miss_idx, col] = np.nan

# A few duplicates
dup_idx = rng.choice(N, size=8, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

# Outliers in monthly spend
out_idx = rng.choice(len(df), size=12, replace=False)
df.loc[out_idx, "monthly_packaging_spend"] = (
    df.loc[out_idx, "monthly_packaging_spend"] * rng.uniform(8, 14, size=12)
).astype(int)

# --- Cleaning ---
df = df.drop_duplicates(subset=["respondent_id"]).reset_index(drop=True)

# Fill categorical NaNs with mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Fill numeric NaNs with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Cap spend outliers at 99th percentile (winsorize)
cap = df["monthly_packaging_spend"].quantile(0.99)
df.loc[df["monthly_packaging_spend"] > cap, "monthly_packaging_spend"] = int(cap)

# ---------------------------------------------------------------------------
# 16. ENGINEERED FEATURES (added in app.py too, but stored here for convenience)
# ---------------------------------------------------------------------------

pain_cols = ["pain_high_cost", "pain_poor_quality", "pain_delayed_delivery",
             "pain_inconsistent_quality", "pain_no_eco_options",
             "pain_poor_customization", "pain_limited_reliability",
             "pain_moq_issue", "pain_lead_time"]
df["supplier_pain_score"] = sum((df[c] == "Yes").astype(int) for c in pain_cols)

barrier_cols = ["barrier_higher_price", "barrier_quality_concern",
                "barrier_current_relationship", "barrier_lack_trusted_suppliers",
                "barrier_supply_inconsistency", "barrier_lack_awareness",
                "barrier_customer_rejection_fear", "barrier_operational_risk"]
df["switching_barrier_score"] = df[barrier_cols].sum(axis=1)

cb_cols = ["cb_samples", "cb_pilot_order", "cb_competitive_pricing",
           "cb_bulk_discount", "cb_quality_warranty", "cb_certification_proof",
           "cb_delivery_reliability", "cb_custom_branding", "cb_flexible_payment"]
df["trust_builder_score"] = sum((df[c] == "Yes").astype(int) for c in cb_cols)

df["sustainability_mindset_score"] = (
    df["sustainability_importance"] + df["plastic_impact_awareness"]
    + df["customer_demand_eco"] + df["esg_image_importance"]
    + df["regulatory_pressure_awareness"]) / 5

pref_cols = ["pref_food_containers", "pref_bowls", "pref_cups", "pref_carry_bags",
             "pref_paper_mailers", "pref_corrugated_boxes", "pref_meal_trays",
             "pref_protective_packaging", "pref_premium_branded"]
df["packaging_demand_score"] = sum((df[c] == "Yes").astype(int) for c in pref_cols)

df["price_sensitivity_score"] = (
    df["barrier_higher_price"] * 1.3
    + df["barrier_supply_inconsistency"] * 0.4
    - df["willingness_to_pay_premium_pct"] / 10)

df["adoption_readiness_score"] = (
    df["sustainability_mindset_score"] * 12
    + df["supplier_pain_score"] * 2.0
    + (df["customer_facing_packaging"] == "Yes").astype(int) * 6
    - df["switching_barrier_score"] * 0.8
    + np.log1p(df["monthly_packaging_spend"]) * 1.2
).round(2)

df["commercial_value_score"] = (
    np.log1p(df["monthly_packaging_spend"]) * 7
    + df["packaging_demand_score"] * 2
    + (df["customer_facing_packaging"] == "Yes").astype(int) * 4
).round(2)

df["operational_risk_score"] = (
    df["barrier_operational_risk"]
    + df["barrier_supply_inconsistency"]
    + df["barrier_quality_concern"]
    + (df["pain_inconsistent_quality"] == "Yes").astype(int) * 1.5)

df["customer_visibility_score"] = (
    (df["customer_facing_packaging"] == "Yes").astype(int) * 3
    + (df["usecase_takeaway"] == "Yes").astype(int)
    + (df["usecase_delivery"] == "Yes").astype(int)
    + (df["usecase_retail"] == "Yes").astype(int)
    + (df["usecase_ecommerce"] == "Yes").astype(int)
    + (df["pref_premium_branded"] == "Yes").astype(int) * 2)

df["product_fit_score"] = (
    df["packaging_demand_score"] * 2
    + df["req_food_safe"] + df["req_leak_proof"]
    + df["req_strength_durability"] + df["req_compostability"])

# Categorical segments
df["customer_value_segment"] = pd.cut(
    df["commercial_value_score"], bins=[-1, 75, 100, 130, 999],
    labels=["Low Value", "Medium Value", "High Value", "Strategic Account"]
).astype(str)

df["adoption_risk_category"] = pd.cut(
    df["adoption_readiness_score"], bins=[-1, 50, 75, 95, 999],
    labels=["Cold Lead", "Warm Lead", "Hot Lead", "Champion Lead"]
).astype(str)

# Final tidy
df = df.reset_index(drop=True)

OUT = "ecoverge_data.csv"
df.to_csv(OUT, index=False)
print(f"Saved {OUT}  ->  {len(df)} rows x {len(df.columns)} columns")
print(f"Likely-adopter rate (Yes): {(df['likely_adopter']=='Yes').mean()*100:.1f} %")
print(f"Avg monthly packaging spend: INR {df['monthly_packaging_spend'].mean():.0f}")
print(f"Avg expected biodegradable budget: INR {df['expected_biodegradable_budget'].mean():.0f}")
