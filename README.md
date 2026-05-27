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

## Run in Google Colab (Recommended)

No local setup needed. Click the badge below to open the notebook directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RpWHF8HSSYryp2ZW1IRRKi2tfQCY-2ik?usp=sharing)


The first cell in the notebook installs all dependencies for you:

```python
!pip install sentence-transformers faiss-cpu scipy scikit-learn -q
```

Just click **Runtime → Run All** and the full pipeline runs end to end in about 2–3 minutes on a free Colab CPU.

---

## Local Installation (Optional)

If you prefer to run locally:

```bash
git clone https://github.com/Kajal2912/RAG-credit-card-recommendation
cd rag-credit-card-recommendation
pip install sentence-transformers faiss-cpu scipy scikit-learn numpy pandas
jupyter notebook research_book3.ipynb
```

---

## What the Notebook Does

Running `research_book3.ipynb` from top to bottom will:

1. Build the 20-card dataset and generate 2,000 synthetic users
2. Build the FAISS embedding index (≈ 0.46 seconds)
3. Run all 4 models (Oracle, UBCF, Cosine-CB, RAG-Hybrid) under all 4 scenarios
4. Run the α ablation (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0})
5. Run paired t-tests across all scenarios and print exact p-values
6. Print a case study with generated explanations for a sample user
7. Save all results to `experiment_results.json`

### Adjusting the α parameter

Inside the notebook, find the alpha ablation cell and change the value:

```python
# Recommended for cold-start scenarios
preds = run_rag_hybrid(test_users, df_cards, model, index, alpha=0.25)

# Pure financial utility (best when user spend profile is available)
preds = run_rag_hybrid(test_users, df_cards, model, index, alpha=0.0)
```

---

## Project Structure

```
rag-credit-card-recommendation/
│
├── Full Pipeline-code.ipynb       ← Full pipeline notebook (open in Colab)
├── experiment_results.json    ← Pre-computed results (all scenarios)
│
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

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{thakur2025ragcredit,
  author    = {Kajal Thakur and Vijay Khadse},
  title     = {Explainable {RAG}-Based Credit Card Recommendation System
               with Financial Utility Optimization and Cold-Start Robustness},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The credit card attribute data is compiled from publicly available product pages of HDFC Bank, ICICI Bank, SBI Cards, and Axis Bank (India). No proprietary transaction data is used. The user dataset is entirely synthetic.

---

## Contact

**Kajal Thakur** — `kajal.thakur@coeptech.ac.in`
Department of Computer Science and Engineering
COEP Technological University, Pune, India
