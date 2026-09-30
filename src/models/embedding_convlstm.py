import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.models.convlstm import ConvLSTMCell

class EmbeddingConvLSTM(nn.Module):
    """
    Amplified ConvLSTM featuring joint representation learning:
    1. Spatial Region Embeddings for grid cells (8x8 = 64 regions)
    2. Crime-Type Category Embeddings (21 crime categories)
    Jointly trained end-to-end with the spatiotemporal risk forecasting task.
    """
    def __init__(self, grid_h=8, grid_w=8, num_crime_types=21, embed_dim=16, hidden_channels=32, num_layers=2):
        super(EmbeddingConvLSTM, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_regions = grid_h * grid_w
        self.num_crime_types = num_crime_types
        self.embed_dim = embed_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Learned Joint Embeddings
        self.region_embedding = nn.Embedding(self.num_regions, embed_dim)
        self.crime_embedding  = nn.Embedding(num_crime_types, embed_dim)

        # Total input channels: 1 (input risk sequence) + embed_dim (spatial region representation map)
        in_channels = 1 + embed_dim

        cell_list = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_in, hidden_channels, kernel_size=3, padding=1))
        self.cell_list = nn.ModuleList(cell_list)

        # Output prediction head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels + embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus()
        )

    def _get_region_map(self, batch_size, device):
        """Builds spatial tensor map of region embeddings (batch, embed_dim, H, W)."""
        region_ids = torch.arange(self.num_regions, device=device) # (64,)
        region_embeds = self.region_embedding(region_ids) # (64, embed_dim)
        region_map = region_embeds.view(self.grid_h, self.grid_w, self.embed_dim).permute(2, 0, 1) # (embed_dim, H, W)
        region_map = region_map.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (batch, embed_dim, H, W)
        return region_map

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 1, H, W)
        returns pred: (batch_size, 1, H, W)
        """
        batch_size, seq_len, _, height, width = x.size()
        device = x.device

        region_map = self._get_region_map(batch_size, device)

        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].init_hidden(batch_size, (height, width), device))

        for t in range(seq_len):
            x_t = x[:, t, :, :, :] # (batch, 1, H, W)
            x_augmented = torch.cat([x_t, region_map], dim=1) # (batch, 1+embed_dim, H, W)
            
            for layer in range(self.num_layers):
                h, c = hidden_states[layer]
                h_next, c_next = self.cell_list[layer](x_augmented if layer == 0 else h, (h, c))
                hidden_states[layer] = (h_next, c_next)

        final_h = hidden_states[-1][0] # (batch, hidden_channels, H, W)
        fused_repr = torch.cat([final_h, region_map], dim=1) # (batch, hidden_channels+embed_dim, H, W)
        pred_lambda = self.head(fused_repr) + 1e-5
        return pred_lambda


def visualize_embeddings_tsne(model, idx_to_crime_type=None, save_path='tsne_embeddings.png'):
    """
    Extracts learned Region & Crime-Type Embedding vectors from model and projects them via t-SNE.
    Saves publication-quality 2D scatter plots to save_path.
    """
    model.eval()
    with torch.no_grad():
        region_vecs = model.region_embedding.weight.cpu().numpy() # (64, embed_dim)
        crime_vecs  = model.crime_embedding.weight.cpu().numpy()  # (21, embed_dim)

    # 1. t-SNE on Region Embeddings
    tsne_reg = TSNE(n_components=2, perplexity=min(15, len(region_vecs)-1), random_state=42)
    reg_2d = tsne_reg.fit_transform(region_vecs)

    # 2. t-SNE on Crime Category Embeddings
    tsne_crime = TSNE(n_components=2, perplexity=min(5, len(crime_vecs)-1), random_state=42)
    crime_2d = tsne_crime.fit_transform(crime_vecs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Learned Joint Representation Embeddings (t-SNE Visualization)', fontsize=16, fontweight='bold')

    # Plot 1: Region Embeddings
    grid_h, grid_w = model.grid_h, model.grid_w
    cell_indices = np.arange(len(region_vecs))
    scatter1 = axes[0].scatter(reg_2d[:, 0], reg_2d[:, 1], c=cell_indices, cmap='viridis', s=120, edgecolors='black', alpha=0.85)
    axes[0].set_title('Spatial Region Grid Embeddings (8x8 cells)', fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    fig.colorbar(scatter1, ax=axes[0], label='Grid Cell Linear Index')
    
    # Annotate sample grid cells
    for i in range(0, len(reg_2d), 8):
        r, c = i // grid_w, i % grid_w
        axes[0].annotate(f'({r},{c})', (reg_2d[i, 0]+0.3, reg_2d[i, 1]+0.3), fontsize=8, alpha=0.9)

    # Plot 2: Crime-Type Embeddings
    axes[1].scatter(crime_2d[:, 0], crime_2d[:, 1], color='#e74c3c', s=150, edgecolors='black', alpha=0.9)
    axes[1].set_title('Crime Category Embeddings', fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    if idx_to_crime_type:
        for idx, ct in idx_to_crime_type.items():
            if idx < len(crime_2d):
                axes[1].annotate(ct, (crime_2d[idx, 0]+0.2, crime_2d[idx, 1]+0.2), fontsize=9, fontweight='medium')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Learned Embeddings t-SNE Plot Saved to '{save_path}'")
