import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt

# Define the custom embedding network
class CustomEmbeddingNet2(nn.Module):
    def __init__(self, input_channels, output_dim, sequence_length):
        super(CustomEmbeddingNet2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * (sequence_length // 4), 128),  # Adjust based on pooling
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Define the simulator
def simulator(theta):
    # Simulate a sequence of length 16
    x = torch.sin(theta * 3.14 * torch.linspace(0, 1, steps=16))  # Length of 16
    return x  # Shape: [Batch Size, Sequence Length]

# Define the prior
prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# Input and output dimensions
sequence_length = 16  # Length of the simulated sequence
output_dim = 8        # Desired dimensionality of the embedding

# Instantiate the custom embedding network
embedding_net = CustomEmbeddingNet2(input_channels=1, output_dim=output_dim, sequence_length=sequence_length)

# Define the density estimator with the custom embedding net
density_estimator = posterior_nn(model="maf", embedding_net=embedding_net)

# Initialize SNPE with the density estimator
inference = SNPE(prior=prior, density_estimator=density_estimator)

# Simulate data
theta = prior.sample((1000,))
x = simulator(theta)

# Train SNPE
inferer = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(inferer)

# Condition on an observation and sample from the posterior
theta_test = torch.tensor([[0.5]])
x_obs = simulator(theta_test)
samples = posterior.sample((1000,), x=x_obs)

# Plot the posterior
plt.hist(samples.numpy(), bins=30, density=True, alpha=0.7, label="Posterior samples")
plt.axvline(theta_test.item(), color='red', linestyle='--', label="True theta")
plt.legend()
plt.show()
