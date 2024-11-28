import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb

wandb.init(
    project="MAF - Gaussian mixture",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

validation_interval = 500  # Run validation every 500 steps
patience = 80  # Early stopping patience (in validation intervals)
best_validation_loss = float("inf")

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
def generate_data(n_samples, train_ratio=0.8):
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    std = 0.5
    data = []
    for center in centers:
        samples = np.random.normal(center, std, size=(n_samples // len(centers), 2))
        data.append(samples)
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    data = torch.tensor(data, dtype=torch.float32)

    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:]
    return train_data, val_data

# Training function
def train(model, train_data, val_data, epochs, lr):
    patience = 50  # Early stopping patience (in validation intervals)
    best_validation_loss = float("inf")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )
    for epoch in tqdm.tqdm(range(epochs)):

        model.train()
        optimizer.zero_grad()
        train_log_probs = model.log_prob(train_data)
        train_loss = -train_log_probs.mean()
        train_loss.backward()
        optimizer.step()
        wandb.log({"Train Loss": train_loss})

        current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
        wandb.log({"learning_rate": current_lr})

        model.eval()
        with torch.no_grad():
            val_log_probs = model.log_prob(val_data)
            val_loss = -val_log_probs.mean()
            scheduler.step(val_loss)
            wandb.log({"Val Loss": val_loss})

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                no_improvement_count = 0  # Reset counter if improvement is seen
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping triggered.")
                    break  
# Plot learned distribution

def target_density(grid):
    """Compute the target density (mixture of Gaussians) on a grid."""
    density = 0
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    std = 0.5
    for center in centers:
        mvn = MultivariateNormal(torch.tensor(center, dtype=torch.float32), 
                                 torch.eye(2) * std**2)
        density += torch.exp(mvn.log_prob(grid))
    density /= len(centers)  # Normalize by number of components
    return density

def plot_comparison(model, grid_size=100):
    """Plot both the learned and target densities."""
    # Generate grid
    x = torch.linspace(-6, 6, grid_size)
    y = torch.linspace(-6, 6, grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    # Compute learned density
    with torch.no_grad():
        log_probs = model.log_prob(grid)
    learned_density = torch.exp(log_probs).reshape(grid_size, grid_size)

    # Compute target density
    target_density_vals = target_density(grid).reshape(grid_size, grid_size)

    # Plot target density
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, target_density_vals, levels=50, cmap='viridis')
    plt.colorbar(label="Density")
    plt.title("Target Density (Mixture of Gaussians)")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Plot learned density
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, learned_density, levels=50, cmap='viridis')
    plt.colorbar(label="Density")
    plt.title("Learned Density (MAF)")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.tight_layout()
    plt.show()

# Hyperparameters
input_dim = 2
hidden_dim = 200
num_flows = 4
epochs = 1000
lr = 1e-3

# Data
n_samples = 2000
train_data, val_data = generate_data(n_samples)

# Model and training
model = MaskedAutoregressiveFlow(input_dim, hidden_dim, num_flows)
train(model, train_data, val_data, epochs, lr)

# Plot learned distribution
plot_comparison(model)