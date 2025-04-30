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
import corner

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
stddev = torch.Tensor([0.1, 0.2, 0.3])

def linear_gaussian(theta):
    return theta - 1.0 + torch.randn_like(theta) * stddev

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
num_epochs = 15

batch_size = 64

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
    return ((2*torch.pi) ** (-3/2)) * (tau ** -3) * torch.exp(-dist_sq / (2 * tau ** 2))  # [batch_size]

def mahalanobis_kernel(x, x_o, tau):
    """
    x: [batch_size, d]
    x_o: [d]
    tau: scalar
    """
    _, d = x.shape
    diff = x - x_o  
    cov = torch.from_numpy(np.cov(x.T.detach().cpu().numpy())).to(dtype=x.dtype, device=x.device)
    inv_cov = torch.linalg.inv(cov)  # [d, d]
    det_cov = torch.linalg.det(cov)
    # Mahalanobis distance squared: xᵢᵀ Σ⁻¹ xᵢ
    dist_sq = torch.einsum('bi,ij,bj->b', diff, inv_cov, diff)  # [batch_size]
    norm_const = ((2 * torch.pi) ** (-d / 2)) * (det_cov ** -0.5) * (tau ** -d)
    kernel_vals = norm_const * torch.exp(-dist_sq / (2 * tau**2))
    return kernel_vals  # [batch_size]

density_estimator = build_nsf(
    dummy_theta,
    dummy_x,
    embedding_net=BasicEmbeddingNet(),
)
optimizer = AdamW(density_estimator.parameters(), lr=1e-3) # initialise pytorch optimiser

for round in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=5000)
    train_loader = DataLoader(TensorDataset(theta, x), batch_size=batch_size, shuffle=True)
    val_theta, val_x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    val_dataset = TensorDataset(val_theta, val_x)
    val_loader = DataLoader(TensorDataset(val_theta, val_x), batch_size=batch_size, shuffle=False)
    best_validation_loss = float("inf")
    no_improvement_count = 0
    patience = 12

    # if round != 0:
    #     gamma = 0.01
    #     def ess_given_tau(tau_value):
    #         tau_tensor = torch.tensor(tau_value, device=x_o.device, dtype=torch.float32)

    #         weight_sum = 0.0
    #         weight_squared_sum = 0.0

    #         for batch_theta, batch_x in train_loader:
    #             batch_theta = batch_theta.to(x_o.device)
    #             batch_x = batch_x.to(x_o.device)

    #             with torch.no_grad():
    #                 log_p_theta = prior.log_prob(batch_theta)
    #                 log_q_theta = proposal.log_prob(batch_theta)
    #                 log_weights = log_p_theta - log_q_theta
    #                 w = torch.exp(log_weights)

    #                 kernel_vals = gaussian_kernel(batch_x, x_o, tau_tensor)
    #                 importance_weights = kernel_vals * w

    #                 weight_sum += importance_weights.sum().item()
    #                 weight_squared_sum += (importance_weights ** 2).sum().item()

    #         ess = (weight_sum ** 2) / weight_squared_sum
    #         return ess - gamma * 5000
    #     sol = root_scalar(ess_given_tau, bracket=[0.5, 1.5], method='bisect')
    #     if sol.converged:
    #         print(f"Found τ: {sol.root:.4f}")
    #         tau = sol.root
    #     else:
    #         print("Root-finding did not converge, using τ:0.5.")  
    #         tau = 0.5

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

    overall_weights = []
    overall_theta = []
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
                        # kernel_value = gaussian_kernel(batch_x, x_o, tau)
                        # kernel_value = torch.ones_like(losses)
                        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
                        weights = weights / weights.sum() 
                        # weights = kernel_value * torch.exp(log_weights)  
                overall_weights.append(weights)
                overall_theta.append(batch_theta)
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
                        # kernel_value = gaussian_kernel(x_val_batch, x_o, tau)
                        # kernel_value = torch.ones_like(losses)
                        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
                        weights = weights / weights.sum()
                        # weights = kernel_value * torch.exp(log_weights) 
                val_loss = (weights * losses).mean()
                epoch_val_loss += val_loss * theta_val_batch.size(0)
        epoch_val_loss /= len(val_dataset)  
        wandb.log({f"val_loss_round_{round+1}": epoch_val_loss, "epoch": epoch})
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_validation_loss:
            best_validation_loss = epoch_val_loss
            no_improvement_count = 0 
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break  
    
    # plot histogram of weights
    all_weights = torch.cat(overall_weights)
    all_theta = torch.cat(overall_theta)
    all_weights = all_weights.cpu().numpy()
    all_theta = all_theta.cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.hist(all_weights, bins=50, density=True)
    plt.yscale("log")
    plt.xlabel("Weights")
    plt.ylabel("Log Density")
    plt.title(f"Histogram of Weights after Round {round+1} with Calibration Kernel")
    plt.xlim(0, max(all_weights)+0.1)
    plt.show()

    # find the index of the maximum weight
    max_weight_index = np.argmax(all_weights)
    min_weight_index = np.argmin(all_weights)
    # get the corresponding theta
    max_theta = all_theta[max_weight_index]
    min_theta = all_theta[min_weight_index]
    posterior = DirectPosterior(density_estimator, prior)
    samples = posterior.sample((10000,), x=x_o)
    posterior_samples.append(samples)
    posterior = posterior.set_default_x(x_o) 
    # proposal = MixtureProposal(posterior, prior, alpha=0.2)
    proposal = posterior
    proposal_samples = proposal.sample(torch.Size((10000,)))
    fig = plt.figure(figsize=(10, 5))
    # plt.suptitle(f"Histogram of Thetas used in Round {round+1} without Calibration Kernel")
    # for idx in range(num_dim):
    #     ax = plt.subplot(3, 1, idx + 1)
    #     ax.hist(all_theta[:, idx], bins=50, density=True)
    #     ax.set_ylabel("Density")
    #     ax.axvline(x=max_theta[idx], color='r', linestyle='--', label='Max Weight Theta')
    #     ax.axvline(x=min_theta[idx], color='g', linestyle='--', label='Min Weight Theta')
    #     ax.set_title(f"Theta {idx+1}")
    #     ax.legend()
    # plt.tight_layout()
    # plt.show()
    fig = corner.corner(
        all_theta,
        labels=[f"Theta {i+1}" for i in range(num_dim)],
        show_titles=False,
        title_kwargs={"fontsize": 8},
        fill_contours=True,
        levels=[0.68, 0.95],
    )
    fig.suptitle(f"Corner Plot of Thetas used in Round {round+1} with Calibration Kernel", fontsize=16)
    axes = np.array(fig.axes).reshape((num_dim, num_dim))
    #plot the maximum weight theta as a scatter point
    for i in range(num_dim):
        for j in range(i):
            ax = axes[i, j]
            ax.scatter(max_theta[j], max_theta[i], color='red', marker='x', label='Max Weight' if i == 1 and j == 0 else "")
            ax.scatter(min_theta[j], min_theta[i], color='blue', marker='o', label='Min Weight' if i == 1 and j == 0 else "")
    axes[1, 0].legend()
    plt.show()
# plot posterior samples
true_means = [1.5, 0.5, 1.0]
true_stds = [0.1, 0.2, 0.3]

fig, ax = pairplot(
    posterior_samples[-1], limits=[[-3, 3], [-3, 3], [-3, 3]], figsize=(5, 5)
)
fig.suptitle("SNPE with Normalized Weights")

for i in range(num_dim):
    diag_ax = ax[i, i]
    xmin, xmax = diag_ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 500)
    true_pdf = norm.pdf(x_vals, loc=true_means[i], scale=true_stds[i])
    true_pdf_scaled = true_pdf / np.max(true_pdf) * np.max(diag_ax.get_ylim())
    diag_ax.plot(x_vals, true_pdf_scaled, color='red', linestyle='--', label='True PDF')

fig.legend(loc="lower left")
plt.show()
