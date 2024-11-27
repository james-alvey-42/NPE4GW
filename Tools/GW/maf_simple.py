import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

class MaskedLinear(nn.Linear):
    """Masked Linear Layer for autoregressive connections."""
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    """Autoregressive network to parameterize the transformation."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out_scale = nn.Linear(hidden_dim, input_dim)
        self.out_shift = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = F.relu(self.hidden(x))
        scale = self.out_scale(h)
        shift = self.out_shift(h)
        scale = torch.tanh(scale)  # Ensures stability
        return scale, shift

class MaskedAutoregressiveFlow(nn.Module):
    """A simple Masked Autoregressive Flow."""
    def __init__(self, input_dim, hidden_dim, num_flows):
        super().__init__()
        self.base_dist = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        self.flows = nn.ModuleList([MADE(input_dim, hidden_dim) for _ in range(num_flows)])

    def forward(self, x):
        """Transforms base samples to target samples."""
        log_det = 0
        z = x
        for flow in self.flows:
            scale, shift = flow(z)
            z = z * torch.exp(scale) + shift
            log_det += scale.sum(-1)
        return z, log_det

    def inverse(self, z):
        """Transforms target samples back to base samples."""
        x = z
        for flow in reversed(self.flows):
            scale, shift = flow(x)
            x = (x - shift) * torch.exp(-scale)
        return x

    def log_prob(self, x):
        """Computes the log-probability of the input under the model."""
        z, log_det = self.forward(x)
        base_log_prob = self.base_dist.log_prob(z)
        return base_log_prob + log_det

# Generate synthetic data: mixture of Gaussians
def generate_data(n_samples):
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    std = 0.5
    data = []
    for center in centers:
        samples = np.random.normal(center, std, size=(n_samples // len(centers), 2))
        data.append(samples)
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

# Training function
def train(model, data, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        log_probs = model.log_prob(data)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Plot learned distribution
def plot_density(model, grid_size=100):
    x = torch.linspace(-6, 6, grid_size)
    y = torch.linspace(-6, 6, grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    with torch.no_grad():
        log_probs = model.log_prob(grid)
    probs = torch.exp(log_probs).reshape(grid_size, grid_size)
    plt.contourf(xx, yy, probs, levels=50, cmap='viridis')
    plt.colorbar(label="Density")
    plt.title("Learned Probability Distribution")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Hyperparameters
input_dim = 2
hidden_dim = 32
num_flows = 2
epochs = 500
lr = 1e-3

# Data
n_samples = 1000
data = generate_data(n_samples)

# Model and training
model = MaskedAutoregressiveFlow(input_dim, hidden_dim, num_flows)
train(model, data, epochs, lr)

# Plot learned distribution
plot_density(model)