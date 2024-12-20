import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt
import wandb

# Coupling layer for a 1D flow, no splitting needed
wandb.init(
    # set the wandb project where this run will be logged
    project="Normalizing Flows",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)


class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x, reverse=False):
        scale = self.scale_net(x)
        translation = self.translate_net(x)
        if not reverse:
            x = x * torch.exp(scale) + translation
        else:
            x = (x - translation) * torch.exp(-scale)
        return x, scale
        return x, scale

    def log_det_jacobian(self, x):
        scale = self.scale_net(x)
        return scale


# Normalizing flow model for 1D distribution
class NormalizingFlow(nn.Module):
    def __init__(self, num_flows):
        super().__init__()
        self.flows = nn.ModuleList([CouplingLayer() for _ in range(num_flows)])
        self.mean = nn.Parameter(torch.tensor(0.0))  # Trainable base mean
        self.std = nn.Parameter(torch.tensor(1.0))  # Trainable base variance
        self.base_dist = D.Normal(self.mean, self.std)

    def forward(self, x):
        log_det_jacobians = 0
        for flow in self.flows:
            x, log_det_jacobian = flow(x)
            log_det_jacobians += log_det_jacobian
        return x, log_det_jacobians

    def inverse(self, z):
        for flow in reversed(self.flows):
            z, _ = flow(z, reverse=True)
        return z

    def log_prob(self, x):
        z, log_det_jacobians = self.forward(x)
        log_base_prob = -0.5 * torch.log(2 * torch.pi * self.std**2) - (
            z - self.mean
        ) ** 2 / (2 * self.std**2)
        return log_base_prob + log_det_jacobians

    def sample(self, num_samples):
        z = self.base_dist.sample(
            (num_samples, 1)
        )  # Ensures shape compatibility for 1D
        x = self.inverse(z)
        return x


# Training function
def train_flow():
    num_flows = 15
    target_dist = D.Normal(-1, 1.2)  # 1D target distribution
    flow = NormalizingFlow(num_flows)
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    num_steps = 5000

    validation_data = target_dist.sample((1000, 1))
    validation_interval = 500  # Run validation every 500 steps
    patience = 5  # Early stopping patience (in validation intervals)
    best_validation_loss = float("inf")
    no_improvement_count = 0

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )

    for step in range(num_steps):
        x = target_dist.sample((128, 1))  # Ensure shape is [batch_size, 1] for 1D
        loss = -flow.log_prob(x).mean()  # Negative log likelihood

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})

        current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
        wandb.log({"learning_rate": current_lr})

        if step % validation_interval == 0:
            with torch.no_grad():
                validation_loss = -flow.log_prob(validation_data).mean().item()
                wandb.log({"val_loss": validation_loss})
            print(
                f"Step {step}: Training Loss = {loss.item()}, Validation Loss = {validation_loss}"
            )
            # Early stopping check
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                no_improvement_count = 0  # Reset counter if improvement is seen
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping triggered.")
                    break  # Stop training if no improvement seen for 'patience' validations
                    # Step the learning rate scheduler based on validation loss
            scheduler.step(validation_loss)

    with torch.no_grad():
        generated_samples = flow.sample(5000).numpy()

    # Plot the generated samples
    plt.figure(figsize=(8, 6))
    plt.hist(
        generated_samples,
        bins=50,
        density=True,
        alpha=0.6,
        color="b",
        label="Generated Samples",
    )
    x = torch.linspace(-5, 7, 1000).unsqueeze(1)
    plt.plot(
        x.numpy(),
        target_dist.log_prob(x).exp().numpy(),
        "r",
        linewidth=2,
        label="Target PDF",
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Samples Generated by Normalizing Flow")
    plt.legend()
    plt.show()


train_flow()
