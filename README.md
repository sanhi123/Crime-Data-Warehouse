# Spatiotemporal Crime Hotspot Prediction

A system that predicts **where and when** crime hotspots are likely to emerge, combining a classical clustering baseline with a deep learning forecasting model — evaluated with rigorous, backtested comparison between the two.

---

## 📌 Overview

Most crime analysis tools identify hotspots by clustering *historical* incident locations. That shows where crime *has happened*, but treats the city as static: a region flagged as high-risk stays flagged indefinitely, regardless of whether patterns have since shifted.

This project asks a different question: **can we predict how crime patterns will evolve over the next time window, using both spatial structure (nearby regions influence each other) and temporal structure (hotspots emerge, intensify, and fade over time)?**

To answer it, the system combines a K-means clustering baseline (with a TF-IDF-based conversational interface for querying incident reports) with a ConvLSTM deep learning model that forecasts risk one week ahead, and evaluates whether the added complexity of the deep learning approach is actually worth it — with real backtested metrics, not just anecdotal improvement.

---

## 🎯 Problem Statement

Given historical crime incident records (timestamp, location, crime type), predict a **risk map for the next time window** at grid-cell resolution, and evaluate how this forecast performs against a static historical-clustering baseline at anticipating where incidents will actually occur.

This is framed as a genuine **model comparison study**, not a single model showcase — the goal is to understand *when* deep learning earns its complexity over simpler methods, and be honest about where it doesn't.

---

## 🏗️ System Components

### 1. Static Clustering Baseline
K-means clustering on historical incident coordinates produces fixed hotspot centroids. TF-IDF + cosine similarity power a chatbot interface for querying incident reports. This baseline has no notion of time — its predictions for next week are identical to its predictions for last month.

### 2. Deep Learning Forecaster — ConvLSTM
Crime data is reshaped into a spatiotemporal tensor: a sequence of weekly grids, where each cell holds incident counts (and auxiliary features like rolling averages and day-of-week distribution). A ConvLSTM — convolutional gates inside an LSTM cell — learns spatial correlation between neighboring cells *and* temporal evolution across weeks simultaneously, and outputs a predicted risk grid for the following week.

### 3. Graph Neural Network Variant (optional)
Grid cells are represented as graph nodes connected by spatial adjacency and by historical correlation in crime patterns (rather than a fixed square neighborhood). A GCRN (graph convolution + GRU) models this more flexible notion of "which regions influence which," and is compared directly against the ConvLSTM.

### 4. Learned Representations
Region identity and crime type are encoded as learned embeddings, trained jointly with the ConvLSTM rather than represented as raw counts or one-hot vectors. A t-SNE projection of the learned region embeddings is used to visually check whether regions with similar crime dynamics cluster together in embedding space — a concrete demonstration of representation learning rather than just prediction accuracy.

### 5. Evaluation Framework
All models are evaluated on a held-out, time-ordered test window (never randomly split — this is a forecasting problem, and future data must never leak into training). Metrics include:
- **Precision@K / Recall@K** — of the top-K predicted highest-risk cells, what fraction of actual incidents fell within them
- **PAI (Predictive Accuracy Index)** — rewards models that concentrate correct predictions into a small area rather than spreading risk broadly
- **Calibration** — do predicted risk scores actually correspond to real incident rates
- **Rolling backtest** — simulated week-by-week deployment across the entire test period, not a single snapshot

### 6. Conversational Interface
A chatbot answers natural-language queries such as "what's the risk forecast for [area] next week" — mapping the queried location to a grid cell, running the trained model, and returning a risk assessment alongside historical context from the baseline.

---

## 📁 Repository Structure

```
crime-hotspot-prediction/
├── data/
│   ├── raw/                      # original incident records (not committed)
│   └── processed/                # generated spatiotemporal tensors (not committed)
├── src/
│   ├── data_pipeline.py          # grid construction, weekly binning, tensor generation
│   ├── baseline.py               # K-means clustering baseline
│   ├── convlstm_model.py         # ConvLSTM architecture and training loop
│   ├── gnn_model.py              # optional GCRN variant
│   ├── embeddings.py             # learned region/crime-type embeddings
│   ├── evaluation.py             # Precision@K, Recall@K, PAI, calibration, backtest
│   └── chatbot_integration.py    # conversational risk-forecast interface
├── notebooks/
│   └── exploration_and_results.ipynb   # full walkthrough, training curves, comparison table
├── results/
│   └── comparison_table.csv
├── requirements.txt
└── README.md
```

---

## 📊 Results & Exploration

The full analysis, model training, and results are documented in
[`notebooks/exploration_and_results.ipynb`](notebooks/exploration_and_results.ipynb).

This notebook covers:
- Exploratory analysis of the crime dataset (temporal and spatial trends)
- Baseline predictions from the K-means clustering pipeline
- ConvLSTM training and validation curves
- Comparison table: baseline vs. ConvLSTM (vs. GNN, if built) across Precision@K, Recall@K, PAI, and backtest results
- t-SNE visualization of learned region embeddings
- Key visualizations, including predicted vs. actual hotspot maps

**Key finding:** *[fill in once you have results — e.g., "ConvLSTM improved recall@10 by X% over the static baseline, with the largest gains in high-density regions; the baseline remained more stable in sparse, low-data regions."]*

| Model | Precision@10 | Recall@10 | PAI | Backtest Avg Precision |
|---|---|---|---|---|
| K-means Baseline (static) | — | — | — | — |
| ConvLSTM | — | — | — | — |
| GNN (optional) | — | — | — | — |

*(Fill in this table once evaluation is complete — this is the centerpiece of the project.)*

---

## ⚙️ Setup & Reproduction

```bash
git clone https://github.com/<your-username>/crime-hotspot-prediction.git
cd crime-hotspot-prediction
pip install -r requirements.txt
```

1. Place raw incident data in `data/raw/` (see `src/data_pipeline.py` for expected columns: `timestamp`, `latitude`, `longitude`, `crime_type`)
2. Run the data pipeline to generate processed tensors:
   ```bash
   python src/data_pipeline.py
   ```
3. Train the ConvLSTM:
   ```bash
   python src/convlstm_model.py
   ```
4. Run evaluation and generate the comparison table:
   ```bash
   python src/evaluation.py
   ```
5. Or open [`notebooks/exploration_and_results.ipynb`](notebooks/exploration_and_results.ipynb) to walk through the full pipeline interactively.

---

## 🧠 Design Decisions

- **Time-based train/val/test split, not random** — this is a forecasting task; a random split would leak future information into training and produce misleadingly optimistic results.
- **Poisson loss instead of MSE** for the ConvLSTM — targets are incident counts, and Poisson NLL is the more principled choice for count data than treating it as a continuous regression target.
- **The static baseline is kept, not discarded** — the point of this project isn't to prove deep learning wins by default, but to measure honestly where it helps and where simpler methods remain competitive.
- **Embeddings over one-hot encodings** — learning region and crime-type representations jointly with the prediction task, rather than hand-encoding them, follows a representation-learning approach that generalizes better and is directly inspectable (via the t-SNE projection).

---

## 🔭 Limitations & Future Work

- Grid resolution is a tradeoff: finer grids give more precise localization but sparser per-cell data, hurting learnability. Current resolution: *[fill in]*
- The model currently forecasts one week ahead; multi-step forecasting (predicting 2-4 weeks out) would be more operationally useful but harder to validate reliably.
- Crime data is subject to reporting bias (not all incidents are reported at equal rates across regions) — predictions reflect *reported* incident patterns, not necessarily true underlying crime rates. This is worth stating explicitly rather than implying the model measures ground truth.
- With more time/data: incorporating external features (weather, local events, socioeconomic indicators) and extending to multi-step forecasting are natural next steps.

---

## 🛠️ Tech Stack

- Python**, **PyTorch** (ConvLSTM, GNN, embeddings)
- PyTorch Geometric** (optional GNN variant)
- scikit-learn** (K-means baseline, TF-IDF)
- NumPy / Pandas** (data pipeline)
- Jupyter** (exploration and results)
