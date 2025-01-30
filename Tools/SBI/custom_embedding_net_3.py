import torch
import torch.nn as nn
from torch.distributions import Uniform
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as sbi_utils
import numpy as np
import matplotlib.pyplot as plt

# Define a simple 3D CNN
class CustomEmbeddingNet3(nn.Module):
    def __init__(self):
        super(CustomEmbeddingNet3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv3D
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # Pooling halves each dimension
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv3D
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)  # Pooling halves each dimension again
        )
        
        # Calculate the output size after the last convolution and pooling
        # Input size: 16x16x16 -> Pooling reduces to 8x8x8 -> Pooling reduces to 4x4x4
        output_size = 4 * 4 * 4 * 32  # Depth x Height x Width x Channels

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(output_size, 128),  # Updated input size
            nn.ReLU(),
            nn.Linear(128, 3)  # Output matches the parameter dimension
        )
    
    def forward(self, x):
        x = self.conv(x)  # Apply convolution and pooling
        x = self.flatten(x)  # Flatten for fully connected layers
        x = self.fc(x)  # Apply fully connected layers
        return x
# Custom simulator
def simulator(theta):
    batch_size = theta.shape[0]
    grid_size = 16  # Dimensions of the 3D grid
    x, y, z = torch.meshgrid(
        torch.linspace(-1, 1, grid_size),
        torch.linspace(-1, 1, grid_size),
        torch.linspace(-1, 1, grid_size),
        indexing="ij",
    )
    x, y, z = x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)  # Add batch dim

    # Extract parameters and add singleton dimensions for broadcasting
    mean_x, mean_y, mean_z = theta[:, 0, None, None, None], theta[:, 1, None, None, None], theta[:, 2, None, None, None]

    # Compute Gaussian blob for each batch
    gaussian = torch.exp(
        -((x - mean_x) ** 2 + (y - mean_y) ** 2 + (z - mean_z) ** 2) 
    )

    # Add noise
    noise = 0.05 * torch.randn_like(gaussian)
    return gaussian + noise
# Define the prior distribution
prior = sbi_utils.BoxUniform(low=torch.tensor([-1.0, -1.0, -1.0]), 
                             high=torch.tensor([1.0, 1.0, 1.0]))
# Instantiate the custom embedding network
embedding_net = CustomEmbeddingNet3()
# Define the density estimator with the custom embedding net
density_estimator = posterior_nn(model="maf", embedding_net=embedding_net)
# Initialize SNPE with the density estimator
inference = SNPE(prior=prior, density_estimator=density_estimator)
num_simulations = 500
theta = prior.sample((num_simulations,))
x = simulator(theta)
# Train SNPE
density_estimator = inference.append_simulations(theta, x).train()
# Build posterior
posterior = inference.build_posterior(density_estimator)
# Define a test parameter set
theta_test = torch.tensor([0.5, -0.5, 0.0])  # Test parameter
x_test = simulator(theta_test.unsqueeze(0))  # Simulate data for this parameter
# Sample from the posterior
samples = posterior.sample((1000,), x=x_test)
# Visualize marginal posterior distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
params = ["mean_x", "mean_y", "mean_z"]
for i in range(3):
    axes[i].hist(samples[:, i].numpy(), bins=30, alpha=0.7, color="blue", density=True)
    axes[i].axvline(theta_test[i].item(), color="red", linestyle="--", label="True")
    axes[i].set_title(f"Posterior of {params[i]}")
    axes[i].legend()

plt.tight_layout()
plt.show()
