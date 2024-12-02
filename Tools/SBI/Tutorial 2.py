import torch
from torch import ones, eye
from torch.optim import Adam, AdamW
from matplotlib import pyplot as plt
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_nsf
from sbi.neural_nets.net_builders import build_maf
from typing import Callable

class NPEData(torch.utils.data.Dataset):

    def __init__(self,
                 num_samples: int,
                 prior: torch.distributions.Distribution,
                 simulator: Callable,
                 seed: int = 44):
        super().__init__()

        torch.random.manual_seed(seed) #will set the seed device wide
        self.prior = prior
        self.simulator = simulator

        self.theta = prior.sample((num_samples,))
        self.x = simulator(self.theta)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index:int):
        return self.theta[index,...], self.x[index,...]

prior = BoxUniform(-3 * ones((2,)), 3 * ones((2,)))

def simulator(theta):
    return theta + torch.randn_like(theta) * 0.1

num_simulations = 2000
theta = prior.sample((num_simulations,))
x = simulator(theta)

density_estimator = build_nsf(theta, x)
posterior = DirectPosterior(density_estimator, prior)

opt = Adam(list(density_estimator.parameters()), lr=5e-4)

for _ in range(200):
    opt.zero_grad()
    losses = density_estimator.loss(theta, condition=x)
    loss = torch.mean(losses)
    loss.backward()
    opt.step()

dummy_data = NPEData(64, prior, simulator, seed=43)
dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=4)
dummy_theta, dummy_x = next(iter(dummy_loader))
maf_estimator = build_maf(dummy_theta, dummy_x)

optw = AdamW(list(maf_estimator.parameters()), lr=5e-4)
num_epochs = 100

for ep in range(num_epochs):
    for idx, (theta_batch, x_batch) in enumerate(dummy_loader):
        optw.zero_grad()
        losses = maf_estimator.loss(theta_batch, condition=x_batch)
        loss = torch.mean(losses)
        loss.backward()
        optw.step()
    if ep % 10 == 0:
        print("last loss", loss.item())

x_o = torch.as_tensor([[1.0, 1.0]])
print(f"Shape of x_o: {x_o.shape}")

samples = posterior.sample((1000,), x=x_o)
print(f"Shape of samples: {samples.shape}")

# let's compare the trained estimator to the NSF from above
samples_maf = maf_estimator.sample((1000,), condition=x_o).detach()
print(f"Shape of samples: {samples_maf.shape}  # Samples are returned with a batch dimension.")
samples_maf = samples.squeeze(dim=1)
print(f"Shape of samples: {samples_maf.shape}     # Removed batch dimension.")

_ = pairplot(samples, limits=[[-3, 3], [-3, 3]], figsize=(6, 6), title="NSF")
_ = pairplot(samples_maf, limits=[[-3, 3], [-3, 3]], figsize=(6, 6), title="MAF")

plt.show()