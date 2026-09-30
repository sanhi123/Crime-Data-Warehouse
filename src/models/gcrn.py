import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def build_grid_graph_adj(grid_h=8, grid_w=8, train_tensor=None):
    """
    Constructs graph adjacency matrix A (N x N) where N = grid_h * grid_w (64 nodes).
    Edges combine 8-neighbor spatial connectivity and historical crime co-occurrence correlation.
    """
    N = grid_h * grid_w
    A_spatial = np.zeros((N, N), dtype=np.float32)

    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        n_idx = nr * grid_w + nc
                        dist = np.sqrt(dr**2 + dc**2)
                        A_spatial[idx, n_idx] = 1.0 / dist

    # Add historical crime correlation edges if train_tensor provided
    A_corr = np.zeros((N, N), dtype=np.float32)
    if train_tensor is not None:
        # train_tensor shape: (T_train, 1, H, W)
        flat_series = train_tensor.reshape(train_tensor.shape[0], N) # (T_train, N)
        corr_matrix = np.corrcoef(flat_series.T) # (N, N)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.maximum(corr_matrix, 0.0) # non-negative correlation
        np.fill_diagonal(corr_matrix, 0.0)
        A_corr = (corr_matrix > 0.3).astype(np.float32) * corr_matrix

    A_combined = A_spatial + 0.5 * A_corr
    np.fill_diagonal(A_combined, 1.0) # Self-loops

    # Degree normalization: D^{-1/2} A D^{-1/2}
    deg = A_combined.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    deg_inv_sqrt[mask] = np.power(deg[mask], -0.5)
    D_mat = np.diag(deg_inv_sqrt)
    A_norm = D_mat @ A_combined @ D_mat

    return torch.tensor(A_norm, dtype=torch.float32)


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (batch_size, N, in_features)
        adj: (N, N)
        """
        # Graph convolution: A_norm @ X @ W
        support = self.linear(x) # (batch_size, N, out_features)
        out = torch.matmul(adj, support) # (batch_size, N, out_features)
        return out


class GCRNCell(nn.Module):
    """GCN-based GRU Cell."""
    def __init__(self, in_features, hidden_dim):
        super(GCRNCell, self).__init__()
        self.gcn_gate = GraphConv(in_features + hidden_dim, 2 * hidden_dim)
        self.gcn_candidate = GraphConv(in_features + hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, adj):
        """
        x: (batch, N, in_features)
        h: (batch, N, hidden_dim)
        """
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gcn_gate(combined, adj))
        r, z = torch.split(gates, self.hidden_dim, dim=-1)

        combined_candidate = torch.cat([x, r * h], dim=-1)
        n = torch.tanh(self.gcn_candidate(combined_candidate, adj))

        h_next = (1 - z) * n + z * h
        return h_next


class GCRN(nn.Module):
    def __init__(self, grid_h=8, grid_w=8, in_features=1, hidden_dim=32, out_features=1, adj_matrix=None):
        super(GCRN, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.N = grid_h * grid_w
        self.hidden_dim = hidden_dim

        if adj_matrix is None:
            adj_matrix = build_grid_graph_adj(grid_h, grid_w)
        self.register_buffer('adj', adj_matrix)

        self.cell = GCRNCell(in_features, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
            nn.Softplus()
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 1, H, W)
        returns pred: (batch_size, 1, H, W)
        """
        batch_size, seq_len, _, H, W = x.size()
        device = x.device

        # Reshape spatial grid to graph nodes N = H*W
        x_node = x.view(batch_size, seq_len, H * W, 1) # (batch, seq_len, N, 1)

        h = torch.zeros(batch_size, self.N, self.hidden_dim, device=device)
        for t in range(seq_len):
            x_t = x_node[:, t, :, :] # (batch, N, 1)
            h = self.cell(x_t, h, self.adj)

        pred_node = self.head(h) + 1e-5 # (batch, N, 1)
        pred_grid = pred_node.view(batch_size, 1, H, W)
        return pred_grid
