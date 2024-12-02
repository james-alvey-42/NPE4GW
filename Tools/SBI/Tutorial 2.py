import torch
from torch import ones, eye
from torch.optim import Adam, AdamW

from sbi.utils import BoxUniform
from sbi.analysis import pairplot
from typing import Callable
from sbi.neural_nets.net_builders import build_nsf


prior = BoxUniform(-3 * ones((2,)), 3 * ones((2,)))

def simulator(theta):
    return theta + torch.randn_like(theta) * 0.1

num_simulations = 2000
theta = prior.sample((num_simulations,))
x = simulator(theta)


density_estimator = build_nsf(theta, x)

