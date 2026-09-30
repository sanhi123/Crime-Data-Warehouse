import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.optim as optim

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import CrimeDataPipeline
from src.models.convlstm import ConvLSTM, PoissonLoss
from src.models.gcrn import GCRN, build_grid_graph_adj
from src.models.embedding_convlstm import EmbeddingConvLSTM, visualize_embeddings_tsne
from src.evaluation import ModelEvaluator
from src.chatbot import ExtendedCrimeBot

def main():
    print("=" * 70)
    print("🚀 SPATIOTEMPORAL CRIME FORECASTING & AI CHATBOT PIPELINE")
    print("=" * 70)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Compute Device: {device}\n")

    # ----------------------------------------------------
    # PHASE 1: Data Pipeline & K-Means Baseline
    # ----------------------------------------------------
    print("📌 PHASE 1 — Data Pipeline & Baseline")
    pipeline = CrimeDataPipeline(db_path='crime_warehouse.db', grid_h=8, grid_w=8)
    pipeline.load_and_preprocess()
    
    (X_tr, Y_tr), (X_v, Y_v), (X_te, Y_te) = pipeline.get_train_val_test_splits(seq_len=8)
    
    kmeans_model, y_pred_kmeans = pipeline.train_kmeans_baseline(X_tr, Y_tr, X_te, Y_te, n_clusters=4)

    # Convert NumPy splits to PyTorch Tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    X_v_t  = torch.tensor(X_v,  dtype=torch.float32).to(device)
    Y_v_t  = torch.tensor(Y_v,  dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    Y_te_t = torch.tensor(Y_te, dtype=torch.float32).to(device)

    poisson_loss_fn = PoissonLoss()

    # ----------------------------------------------------
    # PHASE 2: ConvLSTM Core Deep Learning Model
    # ----------------------------------------------------
    print("\n📌 PHASE 2 — ConvLSTM (Core DL Model)")
    convlstm = ConvLSTM(in_channels=1, hidden_channels=32, out_channels=1, num_layers=2).to(device)
    optimizer = optim.Adam(convlstm.parameters(), lr=0.005)

    epochs = 25
    best_val_loss = float('inf')
    best_convlstm_pred = None

    for ep in range(1, epochs + 1):
        convlstm.train()
        optimizer.zero_grad()
        pred = convlstm(X_tr_t)
        loss = poisson_loss_fn(pred, Y_tr_t)
        loss.backward()
        optimizer.step()

        convlstm.eval()
        with torch.no_grad():
            v_pred = convlstm(X_v_t)
            v_loss = poisson_loss_fn(v_pred, Y_v_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_convlstm_pred = convlstm(X_te_t).cpu().numpy()

        if ep % 5 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f}")

    # ----------------------------------------------------
    # PHASE 3: GNN Variant (GCRN: GCNConv + GRU)
    # ----------------------------------------------------
    print("\n📌 PHASE 3 — GNN Variant (GCRN: GCN + GRU)")
    adj_matrix = build_grid_graph_adj(grid_h=8, grid_w=8, train_tensor=Y_tr).to(device)
    gcrn = GCRN(grid_h=8, grid_w=8, in_features=1, hidden_dim=32, out_features=1, adj_matrix=adj_matrix).to(device)
    gcrn_optimizer = optim.Adam(gcrn.parameters(), lr=0.005)

    best_gcrn_val = float('inf')
    best_gcrn_pred = None

    for ep in range(1, epochs + 1):
        gcrn.train()
        gcrn_optimizer.zero_grad()
        pred = gcrn(X_tr_t)
        loss = poisson_loss_fn(pred, Y_tr_t)
        loss.backward()
        gcrn_optimizer.step()

        gcrn.eval()
        with torch.no_grad():
            v_pred = gcrn(X_v_t)
            v_loss = poisson_loss_fn(v_pred, Y_v_t).item()
            if v_loss < best_gcrn_val:
                best_gcrn_val = v_loss
                best_gcrn_pred = gcrn(X_te_t).cpu().numpy()

        if ep % 5 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f}")

    # ----------------------------------------------------
    # PHASE 3.5: Amplified DL Component (Learned Embeddings)
    # ----------------------------------------------------
    print("\n📌 PHASE 3.5 — Amplified DL Model (Learned Embeddings + t-SNE)")
    amp_model = EmbeddingConvLSTM(
        grid_h=8, grid_w=8,
        num_crime_types=pipeline.num_crime_types,
        embed_dim=16,
        hidden_channels=32
    ).to(device)
    amp_optimizer = optim.Adam(amp_model.parameters(), lr=0.005)

    best_amp_val = float('inf')
    best_amp_pred = None

    for ep in range(1, epochs + 1):
        amp_model.train()
        amp_optimizer.zero_grad()
        pred = amp_model(X_tr_t)
        loss = poisson_loss_fn(pred, Y_tr_t)
        loss.backward()
        amp_optimizer.step()

        amp_model.eval()
        with torch.no_grad():
            v_pred = amp_model(X_v_t)
            v_loss = poisson_loss_fn(v_pred, Y_v_t).item()
            if v_loss < best_amp_val:
                best_amp_val = v_loss
                best_amp_pred = amp_model(X_te_t).cpu().numpy()

        if ep % 5 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f}")

    # Generate t-SNE plot for learned embeddings
    visualize_embeddings_tsne(amp_model, pipeline.idx_to_crime_type, save_path='tsne_embeddings.png')

    # ----------------------------------------------------
    # PHASE 4: Evaluation Suite & Comparative Table
    # ----------------------------------------------------
    print("\n📌 PHASE 4 — Comprehensive Evaluation Suite")
    evaluator = ModelEvaluator(grid_h=8, grid_w=8)

    models_preds = {
        'K-Means Baseline': y_pred_kmeans,
        'ConvLSTM': best_convlstm_pred,
        'GNN (GCRN)': best_gcrn_pred,
        'Amplified ConvLSTM (Embeddings)': best_amp_pred
    }

    metrics_list = []
    for m_name, pred_array in models_preds.items():
        m_dict, _ = evaluator.evaluate_model(m_name, Y_te, pred_array, k=10)
        metrics_list.append(m_dict)

    df_results = evaluator.create_comparison_table(metrics_list)
    print("\n📊 --- FINAL MODEL COMPARISON TABLE ---")
    print(df_results.to_string(index=False))

    # Generate Evaluation Figures
    evaluator.plot_evaluation_suite(Y_te, models_preds, save_path_prefix='eval_')

    # ----------------------------------------------------
    # PHASE 5: Chatbot Integration
    # ----------------------------------------------------
    print("\n📌 PHASE 5 — Chatbot Integration Demo")
    conn = sqlite3.connect('crime_warehouse.db')
    bot = ExtendedCrimeBot(conn, pipeline, trained_model=amp_model.cpu())

    test_queries = [
        "Predict risk for Mumbai next week",
        "What is the crime forecast for Delhi next week?",
        "Predict risk for Bangalore next week",
        "Which city has the highest crime volume?"
    ]

    for q in test_queries:
        print(f"\n💬 User Query: '{q}'")
        res = bot.answer(q)
        print(res)

    conn.close()

    print("\n✅ All 5 Pipeline Phases Completed Successfully!")

if __name__ == '__main__':
    main()
