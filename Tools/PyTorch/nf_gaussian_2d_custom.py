import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch.distributions as D
import wandb
from scipy.stats import gaussian_kde

 # set the wandb project where this run will be logged
wandb.init(
    project="Normalizing Flows - 2D Gaussian",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)
class Gaussian_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean1 = nn.Parameter(torch.tensor(0.0))  # Trainable base mean
        self.log_var1 = nn.Parameter(torch.tensor(0.0))   # Trainable base variance
        self.mean2 = nn.Parameter(torch.tensor(0.0))  # Trainable base mean
        self.log_var2 = nn.Parameter(torch.tensor(0.0))   # Trainable base variance
        self.rho = nn.Parameter(torch.tensor(0.0))
    def log_prob(self, x, y):
        
        return -0.5 * (self.log_var1 + self.log_var2 + torch.log(1-self.rho**2) + ((x - self.mean1) ** 2 / torch.exp(self.log_var1 + 1e-6)+(y - self.mean2) ** 2 / torch.exp(self.log_var2 + 1e-6) - 2*self.rho*((x - self.mean1)*(y-self.mean2)/(torch.exp(0.5*(self.log_var1+self.log_var2+ 1e-6)))))/(1-self.rho**2))

    def sample(self, num_samples):
        mean = torch.tensor([self.mean1.item(), self.mean2.item()], dtype=torch.float32)
        covariance = torch.tensor([[torch.exp(self.log_var1).item(), self.rho*torch.exp(0.5*(self.log_var1+self.log_var2))], 
                                [self.rho*torch.exp(0.5*(self.log_var1+self.log_var2)), torch.exp(self.log_var2).item()]], dtype=torch.float32)
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)
        return mvn.sample((num_samples,))

mean = torch.tensor([-2, 1],dtype=torch.float32)  
covariance = torch.tensor([[4, 0*2*1], 
                           [0*2*1, 1]], dtype=torch.float32)
true_dist = D.MultivariateNormal(mean, covariance_matrix=covariance)

true = true_dist.sample((1000,))
x_true = true[:,0]
y_true = true[:,1]
val = true_dist.sample((1000,))
x_val = val[:,0]
y_val = val[:,1]

nf = Gaussian_2D()
optimizer = optim.Adam(nf.parameters(), lr=0.01)
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )
losses = []
val_losses = []
samples = []
validation_interval = 500  # Run validation every 500 steps
patience = 80  # Early stopping patience (in validation intervals)
best_validation_loss = float("inf")
no_improvement_count = 0
print(nf.state_dict())

for i in tqdm.tqdm(range(10000)):
    nf.train()
    loss = -nf.log_prob(x_true, y_true).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    wandb.log({"loss": loss})

    current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate
    wandb.log({"learning_rate": current_lr})
   
    nf.eval()
    val_loss = -nf.log_prob(x_val, y_val).mean()
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
            samples.append(nf.sample(3000).detach().numpy())
            print("Early stopping triggered.")
            break  # Stop training if no improvement seen for 'patience' validations
            # Step the learning rate scheduler based on validation loss
    if i == 9999:
        samples.append(nf.sample(3000).detach.numpy())

x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = torch.linspace(-10,10,1000).unsqueeze(1)
X, Y = np.meshgrid(x, y)

positions = np.vstack([X.ravel(), Y.ravel()])
samples_array = np.array(samples)
samples_array = np.vstack(samples_array)
kde = gaussian_kde(samples_array.T)
Z = kde(positions).reshape(X.shape)

plt.figure(3, figsize=(8,6))
plt.contourf(X, Y, Z, levels=50, cmap="hot")  # Use contourf for filled heatmap
plt.colorbar(label="Density")
plt.scatter(samples_array[:, 0], samples_array[:, 1], s=5, color='blue', alpha=0.3, label="Data points")  # Overlay points
plt.title("Trained Density Heatmap")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()

plt.show()
