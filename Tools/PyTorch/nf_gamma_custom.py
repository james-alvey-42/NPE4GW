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
    project="Normalizing Flows - Gamma",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

class GammaNF(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0))  # Trainable base scale parameter - degrees of freedom
        self.shape = nn.Parameter(torch.tensor(2.0))
    def log_prob(self, x):
        return  (self.shape - 1)*torch.log(x) -x/(self.scale) - gammaln(self.shape) - (self.shape)*torch.log(self.scale) 
    
    def sample(self, num_samples):
        dist = Gamma(self.shape, 1/self.scale)
        return dist.sample((num_samples,))

gamma_dist = Gamma(3,1/4)
x_true = gamma_dist.sample((1000,))
x_val = gamma_dist.sample((1000,))

nf = GammaNF()
optimizer = optim.Adam(nf.parameters(), lr=0.001)
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )
steps = []
losses = []
val_losses = []
scales, shapes = [], []
samples = []
validation_interval = 500  # Run validation every 500 steps
patience = 800  # Early stopping patience (in validation intervals)
best_validation_loss = float("inf")
no_improvement_count = 0
print(nf.state_dict())
true_log_probs = (3 - 1)*torch.log(x_true) -x_true/(4) - gammaln(torch.tensor(3)) - (3)*torch.log(torch.tensor(3))


for i in tqdm.tqdm(range(10000)):
    steps.append(i)
    nf.train()
    loss = -nf.log_prob(x_true).mean() + true_log_probs.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    wandb.log({"loss": loss})

    current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
    wandb.log({"learning_rate": current_lr})
   
    nf.eval()
    val_loss = -nf.log_prob(x_val).mean() + true_log_probs.mean()
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

    scales.append(nf.scale.item())
    shapes.append(nf.shape.item())
    if i == 9999:
        samples.append(nf.sample(2000).detach().numpy())

true_log_probs = (3 - 1)*torch.log(x_true) -x_true/(4) - gammaln(torch.tensor(3)) - (3)*torch.log(torch.tensor(3))
true_dist = D.Gamma(3,1/4)


if len(steps) > len(scales):
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
plt.plot(steps, scales, label='Scale Parameter')
plt.plot(steps, shapes, label='Shape Parameter')
plt.axhline(-2)
plt.legend()
plt.ylim(-3, 1)

x = torch.linspace(0, 40, 1000).unsqueeze(1)
plt.figure(3)
plt.title("Trained vs Target Distribution")
plt.plot(x.numpy(), true_dist.log_prob(x).exp().numpy(), "r", label="Target PDF")
plt.hist(samples, bins=40,
        density=True,
        alpha=0.6,
        label="Generated Samples")
plt.legend()
plt.show()
