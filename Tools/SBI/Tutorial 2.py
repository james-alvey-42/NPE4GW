import torch
from torch import ones, eye
from torch.optim import Adam, AdamW
from matplotlib import pyplot as plt
from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_nsf


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

x_o = torch.as_tensor([[1.0, 1.0]])
print(f"Shape of x_o: {x_o.shape}")
samples = posterior.sample((1000,), x=x_o)
print(f"Shape of samples: {samples.shape}")

_ = pairplot(samples, limits=[[-3, 3], [-3, 3]], figsize=(3, 3), upper="contour")
plt.show()