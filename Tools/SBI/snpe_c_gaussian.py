import torch

from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform, repeat_rows
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.neural_nets.net_builders import build_nsf
from sbi.inference.posteriors import DirectPosterior, MCMCPosterior
import sys
from torch.optim import AdamW
import torch.nn as nn
import tqdm
import wandb
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from torch.distributions import Distribution
import numpy as np
from scipy.stats import norm

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

class TruncatedPrior(Distribution):
    def __init__(self, base_prior, hpd_intervals):
        """
        base_prior: A torch.distributions.Distribution object.
        hpd_intervals: List of (low, high) bounds for each dimension.
        """
        self.base_prior = base_prior
        self.hpd_intervals = hpd_intervals
        self._event_shape = base_prior.event_shape
        self._batch_shape = torch.Size()
        self.normalization_constant = self._estimate_normalization_constant()

    def _in_hpd(self, theta):
        mask = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)
        for dim, (low, high) in enumerate(self.hpd_intervals):
            mask &= (theta[:, dim] >= low) & (theta[:, dim] <= high)
        return mask

    def _estimate_normalization_constant(self, num_samples=10_000):
        samples = self.base_prior.sample((num_samples,))
        mask = self._in_hpd(samples)
        proportion_in_hpd = mask.float().mean()
        return proportion_in_hpd.clamp(min=1e-8)  # prevent divide-by-zero

    def log_prob(self, theta):
        theta = theta.view(-1, theta.shape[-1])
        base_log_prob = self.base_prior.log_prob(theta)
        mask = self._in_hpd(theta)
        log_norm = torch.log(self.normalization_constant)
        return torch.where(mask, base_log_prob - log_norm, torch.full_like(base_log_prob, -torch.inf))

    def sample(self, sample_shape=torch.Size()):
        num_samples = int(torch.prod(torch.tensor(sample_shape)))
        accepted = []
        while len(accepted) < num_samples:
            samples = self.base_prior.sample((num_samples,))
            mask = self._in_hpd(samples)
            accepted.extend(samples[mask])
        return torch.stack(accepted[:num_samples])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape
                        
wandb.init(project="test_sequential_loops")  
num_dim = 3

prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

def linear_gaussian(theta):
    return theta - 1.0 + torch.randn_like(theta) * 0.3

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(linear_gaussian, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

num_rounds = 4
x_o = torch.zeros(
    3,
)
dummy_theta = torch.randn(64, 3)  # [batch=2, dim=3]
dummy_x = torch.randn(64, 3)      # [batch=2, dim=3]

density_estimator = build_nsf(
        dummy_theta,
        dummy_x,
        embedding_net=BasicEmbeddingNet(),
    )

# Create a validation dataset
posteriors = []
proposal = prior
num_epochs = 10

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
epoch_val_loss = 0.0
posterior_samples =[]

def compute_hpd_interval(samples, alpha=0.05):
    sorted_samples = np.sort(samples)
    n = len(samples)
    interval_index = int(np.floor((1 - alpha) * n))
    intervals = sorted_samples[interval_index:] - sorted_samples[:n - interval_index]

    min_idx = np.argmin(intervals)
    hpd_min = sorted_samples[min_idx]
    hpd_max = sorted_samples[min_idx + interval_index]

    return hpd_min, hpd_max

for round in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=5000)
    train_loader = DataLoader(TensorDataset(theta, x), batch_size=batch_size, shuffle=True)
    val_theta, val_x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    val_dataset = TensorDataset(val_theta, val_x)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_validation_loss = float("inf")
    no_improvement_count = 0
    patience = 4

    optimizer = AdamW(density_estimator.parameters(), lr=1e-3)  

    for epoch in range(num_epochs):
        density_estimator.train()
        total_loss = 0.0
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
            for batch_theta, batch_x in pbar:
                batch_size = batch_theta.size(0)
                num_atoms = batch_size

                # Create contrastive atoms
                probs = torch.ones(batch_size, batch_size, device=batch_theta.device) * (1 - torch.eye(batch_size, device=batch_theta.device)) / (batch_size - 1)
                choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
                contrastive_theta = batch_theta[choices]

                atomic_theta = torch.cat((batch_theta[:, None, :], contrastive_theta), dim=1).reshape(batch_size * num_atoms, -1)
                repeated_x = repeat_rows(batch_x, num_atoms)

                atomic_theta = reshape_to_sample_batch_event(atomic_theta, density_estimator.input_shape)
                repeated_x = reshape_to_batch_event(repeated_x, density_estimator.condition_shape)

                log_prob_prior = prior.log_prob(atomic_theta).reshape(batch_size, num_atoms)
                log_prob_post = density_estimator.log_prob(atomic_theta, repeated_x).reshape(batch_size, num_atoms)

                # Contrastive (SNPE-C style) loss
                unnormalized_log_prob = log_prob_post - log_prob_prior
                log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(unnormalized_log_prob, dim=-1)
                contrastive_loss = -log_prob_proposal_posterior.mean()

                # Combined with non-contrastive (SNPE-A/B style) loss
                theta_pos = reshape_to_sample_batch_event(batch_theta, density_estimator.input_shape)
                x_pos = reshape_to_batch_event(batch_x, density_estimator.condition_shape)
                log_prob_non_atomic = density_estimator.log_prob(theta_pos, x_pos).squeeze(0)

                corrected_non_atomic_loss = -(log_prob_non_atomic).mean()
                loss = contrastive_loss + corrected_non_atomic_loss  # Combined loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                wandb.log({"train_loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "non_atomic_loss": corrected_non_atomic_loss.item()})
                pbar.set_postfix({"Train Loss": f"{loss.item():.4f}| Val Loss: {epoch_val_loss:.4f}| Round: {round+1}"})

        # Validation loop
        density_estimator.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for theta_val_batch, x_val_batch in val_loader:
                batch_size = theta_val_batch.size(0)
                num_atoms = batch_size

                probs = torch.ones(batch_size, batch_size, device=theta_val_batch.device) * (1 - torch.eye(batch_size, device=theta_val_batch.device)) / (batch_size - 1)
                choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
                contrastive_theta = theta_val_batch[choices]

                atomic_theta = torch.cat((theta_val_batch[:, None, :], contrastive_theta), dim=1).reshape(batch_size * num_atoms, -1)
                repeated_x = repeat_rows(x_val_batch, num_atoms)

                atomic_theta = reshape_to_sample_batch_event(atomic_theta, density_estimator.input_shape)
                repeated_x = reshape_to_batch_event(repeated_x, density_estimator.condition_shape)

                log_prob_prior = prior.log_prob(atomic_theta).reshape(batch_size, num_atoms)
                log_prob_post = density_estimator.log_prob(atomic_theta, repeated_x).reshape(batch_size, num_atoms)

                unnormalized_log_prob = log_prob_post - log_prob_prior
                log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(unnormalized_log_prob, dim=-1)
                constastive_val_loss = -log_prob_proposal_posterior.mean()

                 # Combined with non-contrastive (SNPE-A/B style) loss
                theta_pos = reshape_to_sample_batch_event(theta_val_batch, density_estimator.input_shape)
                x_pos = reshape_to_batch_event(x_val_batch, density_estimator.condition_shape)
                log_prob_non_atomic = density_estimator.log_prob(theta_pos, x_pos).squeeze(0)

                corrected_non_atomic_loss = -(log_prob_non_atomic).mean()
                val_loss = contrastive_loss + corrected_non_atomic_loss  # Combined loss
                epoch_val_loss += val_loss.item() * theta_val_batch.size(0)

        epoch_val_loss /= len(val_dataset)
        wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})
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
    posteriors.append(posterior)
    # Find the 2.5 - 97.5% HPD region of the prior
    samples = posterior.sample((10000,), x=x_o)
    posterior_samples.append(samples)
    hpd_intervals = [compute_hpd_interval(samples[:, i], alpha=0.05) for i in range(samples.shape[1])]
    trunc_prior = TruncatedPrior(prior, hpd_intervals)
    trunc_prior, num_parameters, prior_returns_numpy = process_prior(trunc_prior)
    proposal = trunc_prior  

    # proposal = posterior.set_default_x(x_o) #Setting proposal to trained density estimator

# plot posterior samples
true_mean = 1.0
true_std = 0.3

fig, ax = pairplot(
    posterior_samples[1], limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5)
)
fig.suptitle("SNPE-C - Proposal 2")

for i in range(num_dim):
    diag_ax = ax[i, i]
    # Get current x-axis limits
    xmin, xmax = diag_ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 500)
    true_pdf = norm.pdf(x_vals, loc=true_mean, scale=true_std)
    # Rescale for visual alignment with histogram
    true_pdf_scaled = true_pdf / np.max(true_pdf) * np.max(diag_ax.get_ylim())
    diag_ax.plot(x_vals, true_pdf_scaled, color='red', linestyle='--', label='True PDF')

fig.legend(loc="lower left")
plt.show()