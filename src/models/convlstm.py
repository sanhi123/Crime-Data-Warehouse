import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, num_layers=2, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_in, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.cell_list = nn.ModuleList(cell_list)

        # Output prediction head: 1x1 convolution followed by Softplus for Poisson rate parameter lambda
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Softplus() # Enforces positive rate parameter lambda
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, in_channels, H, W)
        returns pred: (batch_size, out_channels, H, W)
        """
        batch_size, seq_len, _, height, width = x.size()
        device = x.device

        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].init_hidden(batch_size, (height, width), device))

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for layer in range(self.num_layers):
                h, c = hidden_states[layer]
                h_next, c_next = self.cell_list[layer](x_t if layer == 0 else h, (h, c))
                hidden_states[layer] = (h_next, c_next)

        # Use final hidden state from top ConvLSTM layer
        final_h = hidden_states[-1][0]
        pred_lambda = self.head(final_h) + 1e-5
        return pred_lambda


class PoissonLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss:
    L(lambda, y) = lambda - y * log(lambda + eps)
    """
    def __init__(self, eps=1e-6):
        super(PoissonLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_lambda, target):
        loss = pred_lambda - target * torch.log(pred_lambda + self.eps)
        return torch.mean(loss)
