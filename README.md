# Supplier Intelligence Platform

**Integrated Risk Assessment & Performance Decision Intelligence for Manufacturing Supply Networks**

## Overview

A unified platform that combines quantitative risk modeling with supplier performance evaluation, enabling manufacturing SMEs to:

- Assess multi-tier supply network risk using Bayesian scoring
- Simulate disruption cascades with SIR epidemic-adapted models
- Quantify financial exposure via Monte Carlo VaR/CVaR analysis
- Evaluate supplier performance with weighted multi-criteria scorecards
- Calculate Total Cost of Ownership including hidden quality and delivery costs
- Compare switching scenarios with payback period analysis

## Architecture

| Module | Method | Purpose |
|---|---|---|
| Risk Dashboard | Bayesian posterior probability (6 evidence signals) | Supplier-level disruption risk scoring |
| Network Analysis | Graph centrality (Degree, Betweenness, PageRank) | Identify single points of failure |
| Scenario Engine | SIR propagation (Monte Carlo, 50+ runs) | Disruption cascade prediction |
| Financial Impact | Monte Carlo simulation (5,000+ iterations) | VaR/CVaR financial exposure |
| Performance Scorecard | Weighted multi-criteria scoring | Supplier ranking by strategic priority |
| Decision Intelligence | TCO analysis (COPQ, delivery cost, switching cost) | Financial trade-off quantification |

## Quick Start

```bash
# Clone or copy project folder
cd supplier-intelligence-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at **http://localhost:8501**

## Project Structure

```
supplier-intelligence-platform/
├── app.py                    # Main Streamlit application (6 tabs)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── SECURITY.md               # Security & data handling documentation
├── models/
│   ├── bayesian_risk.py      # Bayesian posterior risk scoring
│   ├── sir_propagation.py    # SIR cascade propagation model
│   ├── monte_carlo.py        # Monte Carlo financial simulation
│   ├── graph_metrics.py      # Network centrality analysis
│   └── nasa_upgrades.py      # Aerospace PRA techniques (LHS, Weibull, FTA)
├── agents/
│   ├── sentinel.py           # News monitoring & event classification
│   └── orchestrator.py       # Multi-agent coordination
└── data/
    └── sample_network.json   # 12-node synthetic supply network
```

## References

- Tabachová et al. (2024) — SIR propagation in supply networks
- Hosseini & Ivanov (2020) — Bayesian networks in supply chain risk management
- Chopra & Sodhi (2004) — Managing risk to avoid supply chain breakdown
- Brintrup et al. (2021) — Supply network centrality analysis
- Ellram (1995) — Total Cost of Ownership framework
