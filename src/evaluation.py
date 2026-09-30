import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, grid_h=8, grid_w=8):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.total_cells = grid_h * grid_w

    def precision_at_k(self, y_true, y_pred, k=10):
        """
        Computes Precision@K over test batch.
        y_true, y_pred: shape (N_samples, 1, H, W)
        """
        precisions = []
        N_samples = len(y_true)

        for i in range(N_samples):
            yt = y_true[i, 0].flatten()
            yp = y_pred[i, 0].flatten()

            top_k_pred_idx = set(np.argsort(yp)[-k:])
            top_k_true_idx = set(np.argsort(yt)[-k:])

            intersection = len(top_k_pred_idx.intersection(top_k_true_idx))
            precisions.append(intersection / k)

        return float(np.mean(precisions))

    def recall_at_k(self, y_true, y_pred, k=10):
        """
        Computes Recall@K over test batch: fraction of actual top-k high crime cells flagged.
        """
        recalls = []
        N_samples = len(y_true)

        for i in range(N_samples):
            yt = y_true[i, 0].flatten()
            yp = y_pred[i, 0].flatten()

            top_k_pred_idx = np.argsort(yp)[-k:]
            total_actual_crimes_in_k = yt[top_k_pred_idx].sum()
            total_crimes_overall = yt.sum() + 1e-6

            recalls.append(total_actual_crimes_in_k / total_crimes_overall)

        return float(np.mean(recalls))

    def predictive_accuracy_index(self, y_true, y_pred, k=10):
        """
        Predictive Accuracy Index (PAI) = (n / N) / (a / A)
        n: crimes captured in top K area
        N: total actual crimes
        a: top K area (K cells)
        A: total area (64 cells)
        """
        pais = []
        N_samples = len(y_true)
        area_ratio = k / float(self.total_cells)

        for i in range(N_samples):
            yt = y_true[i, 0].flatten()
            yp = y_pred[i, 0].flatten()

            top_k_pred_idx = np.argsort(yp)[-k:]
            n = yt[top_k_pred_idx].sum()
            N = yt.sum() + 1e-6

            hit_rate = n / N
            pai = hit_rate / area_ratio
            pais.append(pai)

        return float(np.mean(pais))

    def calibration_check(self, y_true, y_pred, num_bins=10):
        """
        Computes calibration error by binning predictions into deciles and comparing vs actuals.
        """
        yp_flat = y_pred.flatten()
        yt_flat = y_true.flatten()

        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(yp_flat, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        pred_means, true_means = [], []
        for b in range(num_bins):
            mask = (yp_flat >= bin_edges[b]) & (yp_flat < bin_edges[b+1])
            if mask.sum() > 0:
                pred_means.append(yp_flat[mask].mean())
                true_means.append(yt_flat[mask].mean())

        pred_means = np.array(pred_means)
        true_means = np.array(true_means)
        ece = np.abs(pred_means - true_means).mean() # Expected Calibration Error

        return ece, pred_means, true_means

    def rolling_temporal_backtest(self, y_true, y_pred):
        """Calculates weekly MAE over test sequence."""
        N_samples = len(y_true)
        weekly_maes = []
        for t in range(N_samples):
            mae_t = np.abs(y_true[t] - y_pred[t]).mean()
            weekly_maes.append(mae_t)
        return np.array(weekly_maes)

    def evaluate_model(self, model_name, y_true, y_pred, k=10):
        mae = float(np.abs(y_true - y_pred).mean())
        rmse = float(np.sqrt(np.square(y_true - y_pred).mean()))
        prec_k = self.precision_at_k(y_true, y_pred, k=k)
        rec_k = self.recall_at_k(y_true, y_pred, k=k)
        pai = self.predictive_accuracy_index(y_true, y_pred, k=k)
        ece, p_means, t_means = self.calibration_check(y_true, y_pred)

        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'Precision@10': prec_k,
            'Recall@10': rec_k,
            'PAI@10': pai,
            'Calibration Error (ECE)': ece
        }
        return metrics, (p_means, t_means)

    def create_comparison_table(self, model_metrics_list):
        df_comp = pd.DataFrame(model_metrics_list)
        return df_comp

    def plot_evaluation_suite(self, y_true, model_preds_dict, save_path_prefix='eval_'):
        """
        Generates plots for Calibration curves, Rolling Temporal Backtests, and Performance Comparison.
        """
        # 1. Calibration Plot
        plt.figure(figsize=(9, 6))
        for model_name, y_pred in model_preds_dict.items():
            ece, p_means, t_means = self.calibration_check(y_true, y_pred)
            plt.plot(p_means, t_means, marker='o', linewidth=2, label=f'{model_name} (ECE={ece:.3f})')
        
        # Perfect calibration reference line
        max_val = max([y_pred.max() for y_pred in model_preds_dict.values()])
        plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Calibration')
        plt.title('Model Calibration Check (Predicted vs Observed Crime Rates)', fontweight='bold')
        plt.xlabel('Mean Predicted Crime Rate')
        plt.ylabel('Mean Observed Crime Rate')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig(f'{save_path_prefix}calibration.png', dpi=120, bbox_inches='tight')
        plt.close()

        # 2. Rolling Temporal Backtest Plot
        plt.figure(figsize=(11, 5))
        for model_name, y_pred in model_preds_dict.items():
            weekly_maes = self.rolling_temporal_backtest(y_true, y_pred)
            plt.plot(weekly_maes, label=model_name, linewidth=2)
        
        plt.title('Rolling Temporal Backtest (Weekly Test MAE over Time)', fontweight='bold')
        plt.xlabel('Test Week Index')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig(f'{save_path_prefix}rolling_backtest.png', dpi=120, bbox_inches='tight')
        plt.close()

        print("✅ Evaluation Suite Figures Saved successfully.")
