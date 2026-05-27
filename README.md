# Explainable RAG-Based Credit Card Recommendation System

> **Financial Utility Optimization and Cold-Start Robustness for Indian Fintech**

Official implementation for the paper:

> **"Explainable RAG-Based Credit Card Recommendation System with Financial Utility Optimization and Cold-Start Robustness"**
> Kajal Thakur, Vijay Khadse — COEP Technological University, Pune, India

---

## What This Paper Does

Most credit card recommendation systems use collaborative filtering (CF), which breaks down in two real-world situations:

1. **Item cold-start** — a new card is launched with zero interaction history. CF cannot recommend it.
2. **Financial utility misalignment** — CF learns what is *popular*, not what gives the user the *best financial return*.

This work proposes a **hybrid RAG framework** that combines:
- Dense semantic retrieval (SentenceTransformer + FAISS) — for cold-start robustness
- Explicit financial utility scoring (NetBenefit) — for economic accuracy
- An attribute-grounded explanation module — for verifiable, regulator-suitable rationales

**Key finding from the α ablation:** Pure financial utility scoring (α = 0.0) achieves NDCG@3 = 0.998, while pure semantic retrieval (α = 1.0) achieves only 0.063. This reveals that RAG's primary value in structured financial recommendation is **cold-start robustness**, not ranking quality.

---

## Architecture

```
User Profile
    │
    ▼
Eligibility Filter (Income ≥ Card threshold)
    │
    ▼
Query Construction ──────────────────────────── Card Knowledge Base
    │                                                    │
    ▼                                                    ▼
SentenceTransformer                              Card Embeddings (384-d)
    │                                                    │
    └──────────────► FAISS Retrieval (top-k=8) ◄────────┘
                           │
                           ▼
              Financial Re-ranking (Score_RAG)
              α · sim_cos + (1-α) · norm(NetBenefit)
                           │
                           ▼
                    Top-3 Recommendations
                           │
                           ▼
                   Explanation Module
         (attribute-grounded natural language rationale)
```

Modules highlighted in the paper as **swappable** (ablation-ready):
- Retrieval Module (SentenceTransformer + FAISS)
- Financial Scoring Module (NetBenefit re-ranking)

All other modules are **frozen** across experiments.

---

## Results Summary

### NDCG@3 Across All Scenarios

| Scenario | Oracle | UBCF | Cosine-CB | RAG-Hybrid (α=0.5) |
|---|---|---|---|---|
| Dense | **0.998** | 0.803 | 0.288 | 0.198 |
| Noise 30% | — | 0.783 | 0.266 | 0.198 |
| Weak Cold-Start | — | 0.789 | 0.288 | 0.198 |
| **Strong Cold-Start** | — | **0.415** ↓ | 0.288 | **0.198** ✓ |

> RAG-Hybrid at **α=0.25** achieves NDCG@3 = **0.595** under strong cold-start vs UBCF's 0.415 — a **43.4% relative improvement**.

### α Ablation (Dense Setting)

| α | Precision@3 | Hit Rate@3 | NDCG@3 |
|---|---|---|---|
| 0.00 (pure financial) | 0.998 | 1.000 | **0.998** |
| **0.25 (recommended)** | **0.615** | **0.995** | **0.740** |
| 0.50 | 0.228 | 0.595 | 0.198 |
| 0.75 | 0.120 | 0.328 | 0.090 |
| 1.00 (pure semantic) | 0.100 | 0.290 | 0.063 |

### Statistical Significance (Paired t-test, UBCF vs RAG-Hybrid)

| Scenario | t-statistic | p-value | Sig. |
|---|---|---|---|
| Dense | 47.760 | 4.15 × 10⁻¹⁶⁷ | *** |
| Noise 30% | 46.698 | 8.41 × 10⁻¹⁶⁴ | *** |
| Weak CS | 39.711 | 1.09 × 10⁻¹⁴⁰ | *** |
| Strong CS | 15.499 | 9.55 × 10⁻⁴³ | *** |

---

## Dataset

### Credit Cards (20 cards, 4 issuers)

| Issuer | Cards | Fee Range (Rs) | Income Req. Range (Rs) |
|---|---|---|---|
| HDFC Bank | 5 | 500 – 12,500 | 3,00,000 – 30,00,000 |
| ICICI Bank | 5 | 0 – 12,000 | 2,40,000 – 25,00,000 |
| SBI Cards | 5 | 499 – 10,000 | 1,50,000 – 10,00,000 |
| Axis Bank | 5 | 250 – 50,000 | 1,50,000 – 50,00,000 |

Each card has 11 attributes: annual fee, welcome bonus, reward rates for 5 categories (travel, dining, fuel, shopping, online), income requirement, lounge access, forex markup, fuel surcharge waiver.

### Synthetic Users (2,000 users)

Generated with fixed seed (42) for reproducibility:

```
Income   ~ LogNormal(μ=13.8, σ=0.75), clipped to [Rs 1.5L, Rs 50L]
Spending ~ Uniform(0.2, 0.4) × Income / 12
Category ~ Dirichlet(α=2, 2, 2, 2, 2) across 5 categories
```

---

## Installation

```bash
git clone https://github.com/[your-username]/rag-credit-card-recommendation.git
cd rag-credit-card-recommendation

pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
scipy>=1.9
sentence-transformers>=2.2
faiss-cpu>=1.7
```

> GPU users: replace `faiss-cpu` with `faiss-gpu`

---

## Usage

### Run the full experiment pipeline

```bash
jupyter notebook research_book3.ipynb
```

Or run as a script:

```bash
python full_pipeline.py
```

This will:
1. Generate the 20-card dataset and 2,000 synthetic users
2. Build the FAISS index (≈ 0.46 seconds)
3. Run all 4 models under all 4 scenarios
4. Run the α ablation (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0})
5. Run paired t-tests across all scenarios
6. Print a case study with generated explanations
7. Save all results to `experiment_results.json`

### Run just the RAG model on a custom user

```python
from full_pipeline import (create_card_dataset, calculate_net_benefit,
                            build_rag_index, run_rag_hybrid,
                            generate_explanation)
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load cards and build index
df_cards = create_card_dataset()
model = SentenceTransformer("all-MiniLM-L6-v2")
index, _, _ = build_rag_index(df_cards, model)

# Define a user
user = {
    "user_id": 1,
    "annual_income": 800000,
    "monthly_spend": 20000,
    "travel_spend": 3000,
    "dining_spend": 4000,
    "fuel_spend": 2000,
    "shopping_spend": 7000,
    "online_spend": 4000,
}

# Get recommendations
preds = run_rag_hybrid(pd.DataFrame([user]), df_cards, model, index, alpha=0.25)
scores = pd.Series(preds[1], index=df_cards["card_name"])
top3 = scores.sort_values(ascending=False).head(3).index.tolist()

# Get explanations
nb = {r["card_name"]: calculate_net_benefit(user, r)
      for _, r in df_cards[df_cards["income_requirement"] <= user["annual_income"]].iterrows()}

for card in top3:
    print(generate_explanation(user, card, df_cards, nb))
```

### Adjust the α parameter

```python
# For dense settings with full user profiles → use α=0.0 or α=0.25
preds = run_rag_hybrid(test_users, df_cards, model, index, alpha=0.25)

# For sparse/unknown user profiles → use α=0.75 or α=1.0
preds = run_rag_hybrid(test_users, df_cards, model, index, alpha=0.75)
```

---

## Project Structure

```
rag-credit-card-recommendation/
│
├── research_book3.ipynb       ← Main Jupyter notebook (full pipeline)
├── full_pipeline.py           ← Standalone Python script version
├── experiment_results.json    ← Pre-computed results (all scenarios)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing the Paper Results

All experiments use fixed random seeds for full reproducibility:

| Seed | Used for |
|------|----------|
| `numpy.random.seed(42)` | User generation, train/test split |
| `random.seed(42)` | Oracle construction |
| `random.seed(123)` | Behavioral noise scenario |

Expected output after running the full pipeline:

```
Dataset: 20 cards, 2000 users
FAISS index built in ~0.46s | Dimensions: 384

--- Scenario 1: Dense Deterministic ---
Oracle: P=0.998  HR=1.000  NDCG=0.998
UBCF:   P=0.706  HR=1.000  NDCG=0.803
Cosine: P=0.304  HR=0.570  NDCG=0.288
RAG:    P=0.228  HR=0.595  NDCG=0.198

--- Alpha Ablation (Dense) ---
alpha=0.00: P=0.998  HR=1.000  NDCG=0.998
alpha=0.25: P=0.615  HR=0.995  NDCG=0.740  ← recommended
alpha=0.50: P=0.228  HR=0.595  NDCG=0.198
alpha=0.75: P=0.120  HR=0.328  NDCG=0.090
alpha=1.00: P=0.100  HR=0.290  NDCG=0.063

--- Statistical Tests ---
  Dense       : t(399) =  47.760,  p = 4.15e-167  ***
  Noise-30%   : t(399) =  46.698,  p = 8.41e-164  ***
  Weak CS     : t(399) =  39.711,  p = 1.09e-140  ***
  Strong CS   : t(399) =  15.499,  p = 9.55e-43   ***
```

---

## Sample Explanation Output

For a user with annual income Rs 8,87,622 and primary spending in shopping (Rs 6,486/month):

```
RAG-Hybrid (α=0.25) Top-3 Recommendations:

1. [ICICI Sapphiro | ICICI]
   Primary category: shopping (Rs 6,486/mo).
   Shopping reward: 2.0%. Est. annual reward: Rs 4,293.
   Annual fee: Rs 6,500. Net benefit: Rs 10,793/year.
   Also offers 4.0% travel reward for your Rs 3,628/mo travel spend.

2. [Axis Flipkart | Axis]
   Primary category: shopping (Rs 6,486/mo).
   Shopping reward: 5.0% — highest in eligible catalog.
   Est. annual reward: Rs 5,835. Annual fee: Rs 500.
   Net benefit: Rs 5,835/year.

3. [HDFC Millennia | HDFC]
   Primary category: shopping (Rs 6,486/mo).
   Online reward: 5.0%. Est. annual reward: Rs 5,695.
   Annual fee: Rs 1,000. Net benefit: Rs 5,695/year.
```

Every number in the explanation is directly traceable to the card's attribute document — no hallucination possible.

---

## License

This project is released under the [MIT License](LICENSE).

The credit card attribute data is compiled from publicly available product pages of HDFC Bank, ICICI Bank, SBI Cards, and Axis Bank (India). No proprietary transaction data is used. The user dataset is entirely synthetic.

---

## Contact

**Kajal Thakur** — `kajal.thakur@coeptech.ac.in`
