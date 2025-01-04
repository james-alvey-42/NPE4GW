import torch
import torch.nn as nn
from torch.distributions import Uniform
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as sbi_utils
import numpy as np
import matplotlib.pyplot as plt

class CustomEmbeddingNet4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomEmbeddingNet4, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,    # Feature size of the input (e.g., 1 for [1, 20, 1])
            hidden_size=hidden_size, # Number of hidden units in the LSTM
            batch_first=True         # Input is [Batch, Sequence Length, Features]
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size) # Output size matches parameter dimensions
        )
    
    def forward(self, x):
        # Pass data through LSTM
        _, (h_n, _) = self.lstm(x)  # h_n: Last hidden state, shape [1, Batch, Hidden Size]
        x = h_n[-1]  # Extract last hidden state for the batch, shape [Batch, Hidden Size]
        x = self.fc(x)  # Pass through fully connected layers
        return x
    
def simulator(theta):
    """
    Simulates sequential data based on input parameters theta.
    Args:
        theta: Torch tensor of shape [num_samples, 3].
    Returns:
        Torch tensor of shape [num_samples, sequence_length, features].
    """
    batch_size = theta.shape[0]
    sequence_length = 20
    time = torch.linspace(0, 1, sequence_length)
    
    # Generate sequences based on theta
    mean_x, mean_y, mean_z = theta[:, 0], theta[:, 1], theta[:, 2]
    sequences = (
        mean_x.unsqueeze(1) * torch.sin(2 * torch.pi * time)
        + mean_y.unsqueeze(1) * torch.cos(2 * torch.pi * time)
        + 0.1 * torch.randn(batch_size, sequence_length)  # Add noise
    )
    return sequences.unsqueeze(-1)  # Add feature dimension

prior = sbi_utils.BoxUniform(low=torch.tensor([-1.0, 1.0, 0.1]), 
                             high=torch.tensor([1.0, 10.0, 2.0]))

# Instantiate the embedding network
input_size = 1        # Feature size from the simulator ([Batch, Sequence Length, 1])
hidden_size = 128     # LSTM hidden state size
output_size = 3       # Dimensionality of theta (parameters)
embedding_net = CustomEmbeddingNet4(input_size, hidden_size, output_size)

density_estimator = posterior_nn(model="maf", embedding_net=embedding_net)
inference = SNPE(prior=prior, density_estimator=density_estimator)

num_simulations = 500
theta = prior.sample((num_simulations,))
x = simulator(theta)

density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
# Define a test parameter set
theta_test = torch.tensor([0.5, 7.5, 0.8])  # Test parameter
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

