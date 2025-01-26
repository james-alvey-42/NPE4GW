import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt


# Define the custom embedding network
class CustomEmbeddingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomEmbeddingNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Define the simulator
def simulator(theta):
    return theta + torch.randn_like(theta)  # Example noisy simulator

# Define the prior
prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# Define the embedding net with correct input dimensions
input_dim = 1  # Simulator outputs 1D data
hidden_dim = 32
output_dim = 16
embedding_net = CustomEmbeddingNet(input_dim, hidden_dim, output_dim)

# Define the density estimator using the custom embedding network
posterior_nn_model = posterior_nn(model="maf", embedding_net=embedding_net)

# Set up SNPE inference
inference = SNPE(prior=prior, density_estimator=posterior_nn_model)

# Simulate data
theta = prior.sample((10000,))  # Sample 1000 parameter sets
x = simulator(theta)
x = x.view(-1, 1)  # Ensure shape matches the input_dim of the embedding net

# Train the density estimator
inferer = inference.append_simulations(theta, x).train()

# Build the posterior using the trained density estimator
posterior = inference.build_posterior(inferer)

#Plot the posterior
# Sample from the posterior
theta_test = torch.tensor([[0.3]])  # Example observed data point
x_obs = simulator(theta_test)
samples = posterior.sample((10000,), x=x_obs)

# Plot the posterior
plt.figure(figsize=(8, 5))
plt.hist(samples.numpy(), bins=30, density=True, alpha=0.7, color='blue', label="Posterior samples")
plt.axvline(x=theta_test.item(), color='red', linestyle='--', label="True parameter value")
plt.xlabel("Theta")
plt.ylabel("Density")
plt.title("Posterior Distribution")
plt.legend()
plt.show()