# SAC-LSTM AU Racing — Reinforcement Learning for Pre-Off Betfair Trading

This repository implements a full reinforcement-learning research stack for **Australian horse-racing pre-off trading**, built around a **Soft Actor–Critic (SAC) + LSTM** architecture.  
It enables offline simulation, online paper-trading, and live deployment on Betfair’s exchange (AU region), with deterministic data lineage and reproducible experiments.

---

## 🧭 Objectives

**Primary**
1. **All-green outcome:** maximise the probability that all runners have non-negative liability before the market turns in-play.  
2. **Capital preservation:** minimise worst-case ROI drawdowns through exposure limits and loss caps.

**Secondary**
- Improve mark-to-market (MTM) ROI at T-0 while keeping all-green stability.
- Optimise execution quality (fill rate, slippage, order discipline).
- Ensure robustness under latency, delayed starts, and market churn.

---

## ⚙️ Architecture Overview

The project follows a **Medallion data architecture** and a modular code layout.

### Data layers
| Layer | Description | Example path |
|-------|--------------|--------------|
| **Bronze** | Canonical raw Betfair snapshots, outcomes, orders. | `data/bronze/order_book/…` |
| **Silver** | Cleaned, aligned, feature-engineered timeseries. | `data/silver/runner_ts/…` |
| **Gold** | RL-ready rollouts / episodes with rewards & done flags. | `data/gold/episodes/…` |

### Code modules
```
src/saclstm_au/
├─ io/           # Data ingest, API clients, parquet writers
├─ etl/          # Transformations: Bronze → Silver → Gold
├─ envs/         # Custom Gymnasium environments & wrappers
├─ exec/         # Paper/live trading adapters & risk logic
├─ training/     # SB3 training loops, callbacks, evaluators
├─ policies/     # Custom SAC-LSTM / TQC / QR variants
├─ metrics/      # Logging, tracking, and plotting utilities
└─ utils/        # Config schemas, alignment helpers, etc.
```

### Experiment & artifact layout
```
experiments/     # One folder per training run (metrics, charts, config freeze)
artifacts/       # Saved models, agents, replays, and exports
configs/         # YAML configs for envs, rewards, models, eval, execution
logs/            # Training and pipeline logs
notebooks/       # Jupyter / Colab analysis playbooks
```

---

## 🧩 Key Components

### 1. Environment — `AUPreOffEnvV7TradeCost`
A custom Gym-style environment that simulates pre-off Betfair markets:
- Tick-size-aware continuous price ladder.
- Commission, latency, and slippage modelling.
- Rejection of unaffordable orders (no hard bankrupt).
- Episode termination at in-play or exposure limit breach.
- Reward shaping for all-green, worst-ROI, and MTM ROI.

### 2. Agent — SAC-LSTM & extensions
Default policy: **Soft Actor–Critic with LSTM memory**  
Optional upgrades:
- **TQC-LSTM** (Truncated Quantile Critics) for tail-risk control.  
- **QR-SAC-LSTM** (Quantile Regression) for CVaR-aware objectives.  
- **Auxiliary predictive heads** for short-horizon price/volume/fill forecasts.

### 3. Offline/Online pipelines
- **Ingest**: `cli/ingest.py` — fetch catalogue/order-book data → Bronze.  
- **Transform**: `cli/build_silver.py`, `cli/build_gold.py`.  
- **Train**: `cli/train.py` — orchestrates SB3 training loop, metrics CSV logger.  
- **Evaluate**: `cli/eval.py` — deterministic suite for checkpoint selection.  
- **Paper-trade**: `cli/paper_trade.py` — live simulation with risk caps.

---

## 📊 Metrics & Evaluation

Each evaluation checkpoint logs:
| Metric | Description |
|---------|-------------|
| `green_rate` | % of markets with all-green book at suspend |
| `worst_roi_p95` | 95th percentile worst ROI (tail risk) |
| `mtm_roi_mean` | Mean mark-to-market ROI at T-0 |
| `orders_submitted / filled` | Order discipline |
| `slippage_bps`, `latency_ms` | Execution quality |
| `cvar_5pc` | Conditional Value-at-Risk from quantile critics (if enabled) |

Metrics are written to `experiments/<run>/metrics.csv` and visualised through `metrics/charts.py`.

---

## 🚀 Quickstart

### 1. Setup environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # or poetry install
```

### 2. Configure credentials
Create `betfair_login.json`:
```json
{
  "username": "myusername",
  "password": "mypassword",
  "use_live": false,
  "delayed_app_key": "xxxx-xxxx",
  "live_app_key": "yyyy-yyyy",
  "cert_path": "./client-2048.crt",
  "key_path": "./client-2048.key"
}
```

### 3. Ingest data
```bash
python -m src.cli.ingest --login-json betfair_login.json --since 2025-01-01
```

### 4. Build features
```bash
python -m src.cli.build_silver
python -m src.cli.build_gold
```

### 5. Train agent
```bash
python -m src.cli.train --config configs/model.sac_lstm.small.yaml
```

### 6. Evaluate & visualise
```bash
python -m src.cli.eval --checkpoint artifacts/models/sac_lstm/best.zip
python -m src.saclstm_au.metrics.charts experiments/exp_*/metrics.csv
```

---

## 🧠 Research roadmap

| Phase | Focus | Description |
|-------|--------|-------------|
| **1** | Baseline SAC-LSTM | End-to-end working pipeline with reward shaping & CSV logging |
| **2** | Distributional critics | TQC/QR variants for tail-risk control |
| **3** | Predictive side-car | Supervised ΔLTP/volume/fill models feeding env |
| **4** | Multi-task SAC | Auxiliary forecasting heads on shared LSTM |
| **5** | Paper-trading harness | Safe live test with real Betfair endpoint |
| **6** | Deployment | Automated scheduling, monitoring, and policy selection |

---

## 🧾 Configuration system

All configs are YAML and validated via `pydantic` schemas.

| File | Purpose |
|------|----------|
| `configs/pipeline.yaml` | Data ingest & ETL control |
| `configs/env.au_preoff.yaml` | Environment parameters (commission, latency, etc.) |
| `configs/reward.shaping.yaml` | Reward weights & penalties |
| `configs/model.sac_lstm.small.yaml` | Network and hyperparameters |
| `configs/eval.criteria.yaml` | Checkpoint selection rules |
| `configs/execution.paper.yaml` | Paper/live trading risk limits |

---

## 📁 Directory reference

```
sac_lstm_au_racing/
├── configs/            # YAML configs
├── data/               # Bronze / Silver / Gold layers
├── artifacts/          # Saved models, agents, charts
├── experiments/        # One folder per run
├── logs/               # Pipeline & training logs
├── notebooks/          # Analysis notebooks
├── src/                # Source code (see structure above)
└── tests/              # Unit tests
```

---

## 🔍 Dependencies

- Python ≥ 3.10  
- Stable-Baselines3  
- Gymnasium  
- PyTorch ≥ 2.1  
- pandas, numpy, pyarrow  
- matplotlib, seaborn (optional for charts)  
- pyyaml, pydantic  
- requests, betfairlightweight

---

## 🧩 Extending the project

- **Add new environments:** drop files into `src/saclstm_au/envs/`.
- **Add features:** modify `etl/to_silver.py` and `etl/to_gold.py`.
- **Try new agents:** extend `src/saclstm_au/policies/` and register in `training/train_sac_lstm.py`.
- **Custom metrics:** add to `metrics/trackers.py` and log via callbacks.

---

## 🧱 License & attribution

This repository is owned and maintained by **Projected Construction Group Pty Ltd** (Melbourne, VIC).  
All code and templates are provided for internal research and evaluation use only.  
No warranty is expressed or implied; trading carries financial risk.

© 2025 Projected Construction Group Pty Ltd. All rights reserved.
