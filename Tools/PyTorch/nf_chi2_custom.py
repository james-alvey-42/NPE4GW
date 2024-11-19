import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch.distributions as D
import wandb
from torch.special import gammaln
from torch.distributions.gamma import Gamma


 # set the wandb project where this run will be logged
wandb.init(
    project="Normalizing Flows - Chi Squared",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

class Chi2NF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(3.0))  # Trainable base mean - degrees of freedom

    def log_prob(self, x):
        return  -0.5 * self.mean - gammaln(0.5*self.mean) + (0.5*self.mean - 1)*torch.log(x) - 0.5*x 
    
    def sample(self, num_samples):
        dist = Gamma(0.5*self.mean, 2)
        return dist.sample((num_samples,))

gamma_dist = Gamma(2,2)
x_true = gamma_dist.sample((1000,))
x_val = gamma_dist.sample((1000,))

nf = Chi2NF()
optimizer = optim.Adam(nf.parameters(), lr=0.001)
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )
steps = []
losses = []
val_losses = []
means = []
samples = []
validation_interval = 500  # Run validation every 500 steps
patience = 80  # Early stopping patience (in validation intervals)
best_validation_loss = float("inf")
no_improvement_count = 0
print(nf.state_dict())

for i in tqdm.tqdm(range(10000)):
    steps.append(i)
    nf.train()
    loss = -nf.log_prob(x_true).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    wandb.log({"loss": loss})

    current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
    wandb.log({"learning_rate": current_lr})
   
    nf.eval()
    val_loss = -nf.log_prob(x_val).mean()
    val_losses.append(val_loss.item())
    scheduler.step(val_loss)
    wandb.log({"val_loss": val_loss})

    # Early stopping check
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        no_improvement_count = 0  # Reset counter if improvement is seen
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            samples.append(nf.sample(2000).detach().numpy())
            print("Early stopping triggered.")
            break  # Stop training if no improvement seen for 'patience' validations
            # Step the learning rate scheduler based on validation loss

    means.append(nf.mean.item())
    if i == 9999:
        samples.append(nf.sample(2000).detach().numpy())

true_log_probs = -0.5 * 4 - gammaln(torch.tensor(0.5*4)) + (0.5*4 - 1)*torch.log(x_true) - 0.5*x_true
true_dist = D.Gamma(2,2)

if len(steps) > len(means):
    steps.pop()
    losses.pop()
    val_losses.pop()

plt.figure(1)
plt.title('Loss')
plt.semilogx(steps, losses, label='Train')
plt.plot(steps, val_losses, label='Validation')
plt.axhline(-true_log_probs.mean(), label='True log-prob')
plt.ylim(0.,)
plt.legend(loc='center left')

plt.figure(2)
plt.title("Convergence")
plt.plot(steps, means, label='Mean')
plt.axhline(-2)
plt.axhline(np.log(1.0), label="Truth")
plt.legend()
plt.ylim(-3, 1)

x = torch.linspace(0, 20, 1000).unsqueeze(1)
plt.figure(3)
plt.title("Trained vs Target Distribution")
plt.plot(x.numpy(), true_dist.log_prob(x).exp().numpy(), "r", label="Target PDF")
plt.hist(samples, bins=40,
        density=True,
        alpha=0.6,
        label="Generated Samples")
plt.show()
