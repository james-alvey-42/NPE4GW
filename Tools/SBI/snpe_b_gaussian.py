import torch

from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.neural_nets.net_builders import build_nsf
from sbi.inference.posteriors import DirectPosterior
import sys
from torch.optim import AdamW
import torch.nn as nn
import tqdm
import wandb
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy.optimize import root_scalar

class BasicEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 3, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MixtureProposal(torch.distributions.Distribution):
    def __init__(self, posterior, defensive, alpha):
        super().__init__(validate_args=False)
        self.posterior = posterior
        self.defensive = defensive
        self.alpha = alpha

    def sample(self, sample_shape=torch.Size()):
        # Flatten sample shape to get total number of samples
        n_samples = int(torch.tensor(sample_shape).prod().item()) if len(sample_shape) > 0 else 1

        mask = torch.rand(n_samples) < self.alpha
        n_def = mask.sum().item()
        n_post = n_samples - n_def

        samples = []
        if n_post > 0:
            samples_post = self.posterior.sample((n_post,))
            samples.append(samples_post)
        if n_def > 0:
            samples_def = self.defensive.sample((n_def,))
            samples.append(samples_def)

        all_samples = torch.cat(samples, dim=0)

        # Shuffle to avoid ordering artifacts
        indices = torch.randperm(n_samples)
        mixed_samples = all_samples[indices]

        # Reshape to match the requested sample shape
        return mixed_samples.view(*sample_shape, -1)

    def log_prob(self, x):
        log_p = self.posterior.log_prob(x)
        log_d = self.defensive.log_prob(x)
        log_mix = torch.logaddexp(
            torch.log(torch.tensor(1 - self.alpha)) + log_p,
            torch.log(torch.tensor(self.alpha)) + log_d
        )
        return log_mix


wandb.init(project="test_sequential_loops")  
num_dim = 3

prior = BoxUniform(low=-3 * torch.ones(num_dim), high=3 * torch.ones(num_dim))

def linear_gaussian(theta):
    return theta - 1.0 + torch.randn_like(theta) * 0.3

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(linear_gaussian, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

num_rounds = 4
x_o = torch.Tensor([0.5, -0.5, 0])  
dummy_theta = torch.randn(64, 3)  
dummy_x = torch.randn(64, 3)      

density_estimator = build_nsf(
        dummy_theta,
        dummy_x,
        embedding_net=BasicEmbeddingNet(),
    )

# Create a validation dataset
proposal = prior
num_epochs = 30

batch_size = 64
optimizer = AdamW(density_estimator.parameters(), lr=1e-3) # initialise pytorch optimiser
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=4,
        verbose=True,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    )
posterior_samples =[]
epoch_val_loss = 0.0
tau = 0.1

def gaussian_kernel(x, x_o, tau):
    """
    x: Tensor of shape [batch_size, dim]
    x_o: Tensor of shape [dim]
    tau: float
    Returns: Tensor of shape [batch_size]
    """
    diff = x - x_o  # [batch_size, dim]
    dist_sq = torch.sum(diff ** 2, dim=1)  # [batch_size]
    return ((2*torch.pi) ** (-3/2)) * (tau ** -3) *torch.exp(-dist_sq / (2 * tau ** 2))  # [batch_size]


for round in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=5000)
    train_loader = DataLoader(TensorDataset(theta, x), batch_size=batch_size, shuffle=True)
    val_theta, val_x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    val_dataset = TensorDataset(val_theta, val_x)
    val_loader = DataLoader(TensorDataset(val_theta, val_x), batch_size=batch_size, shuffle=False)
    best_validation_loss = float("inf")
    no_improvement_count = 0
    patience = 12

    if round != 0:
        gamma = 0.01
        def ess_given_tau(tau_value):
            tau_tensor = torch.tensor(tau_value, device=x_o.device, dtype=torch.float32)
            
            weights = torch.tensor(0.0, device=x_o.device)
            weights_squared = torch.tensor(0.0, device=x_o.device)

            for batch_theta, batch_x in train_loader:
                log_p_theta = prior.log_prob(batch_theta)
                log_q_theta = proposal.log_prob(batch_theta)
                log_weights = log_p_theta - log_q_theta

                kernel_value = gaussian_kernel(batch_x, x_o, tau_tensor)
                importance_weights = kernel_value * torch.exp(log_weights)

                weights += importance_weights.sum()
                weights_squared += (importance_weights ** 2).sum()

            ess = (weights ** 2 / weights_squared).item()
            return ess - gamma*5000
        
        sol = root_scalar(ess_given_tau, bracket=[0.01, 0.5], method='bisect')
        if sol.converged:
            print(f"Found τ: {sol.root:.4f}")
            tau = sol.root
        else:
            print("Root-finding did not converge, using τ:0.5.")  
            tau = 0.5

    optimizer = AdamW(density_estimator.parameters(), lr=1e-3) # initialise pytorch optimiser

    for epoch in range(num_epochs):
        density_estimator.train()
        total_loss = 0.0
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
            for batch_theta, batch_x in pbar:
                losses = density_estimator.loss(batch_theta, batch_x) 
                if round == 0:
                    log_weights = torch.zeros_like(losses)
                    weights = torch.exp(log_weights)  
                else: 
                    with torch.no_grad():
                        log_p_theta = prior.log_prob(batch_theta)
                        log_q_theta = proposal.log_prob(batch_theta)
                        log_weights = log_p_theta - log_q_theta
                        kernel_value = gaussian_kernel(batch_x, x_o, tau)
                        # kernel_value = torch.ones_like(losses)
                        weights = kernel_value * torch.exp(log_weights)  
                loss = (weights * losses).mean() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log({"train_loss": loss.item()})
                pbar.set_postfix({"Train Loss": f"{loss.item():.4f}| Val Loss: {epoch_val_loss:.4f}| Round: {round+1}"})

        density_estimator.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for theta_val_batch, x_val_batch in val_loader:
                losses = density_estimator.loss(theta_val_batch, x_val_batch)
                if round == 0:
                    log_weights = torch.zeros_like(losses)
                    weights = torch.exp(log_weights)  
                else:
                    with torch.no_grad():
                        log_p_theta = prior.log_prob(theta_val_batch)
                        log_q_theta = proposal.log_prob(theta_val_batch)
                        log_weights = log_p_theta - log_q_theta
                        kernel_value = gaussian_kernel(x_val_batch, x_o, tau)
                        # kernel_value = torch.ones_like(losses)
                        weights = kernel_value * torch.exp(log_weights)  
                val_loss = (weights * losses).mean()
                epoch_val_loss += val_loss * theta_val_batch.size(0)
        epoch_val_loss /= len(val_dataset)  
        wandb.log({f"val_loss_{round}": epoch_val_loss, "epoch": epoch})
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_validation_loss:
            best_validation_loss = epoch_val_loss
            no_improvement_count = 0 
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break  

    posterior = DirectPosterior(density_estimator, prior)
    samples = posterior.sample((10000,), x=x_o)
    posterior_samples.append(samples)
    posterior = posterior.set_default_x(x_o) #Setting proposal to trained density estimator
    proposal = MixtureProposal(posterior, prior, alpha=0.2)
    
# plot posterior samples
true_means = [1.5, 0.5, 1.0]
true_std = 0.3

fig, ax = pairplot(
    posterior_samples[-1], limits=[[-1, 3], [-2, 2], [-2, 2]], figsize=(5, 5)
)
fig.suptitle("SNPE with Adaptive Calibration Kernels")

for i in range(num_dim):
    diag_ax = ax[i, i]
    xmin, xmax = diag_ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 500)
    true_pdf = norm.pdf(x_vals, loc=true_means[i], scale=true_std)
    true_pdf_scaled = true_pdf / np.max(true_pdf) * np.max(diag_ax.get_ylim())
    diag_ax.plot(x_vals, true_pdf_scaled, color='red', linestyle='--', label='True PDF')

fig.legend(loc="lower left")
plt.show()
