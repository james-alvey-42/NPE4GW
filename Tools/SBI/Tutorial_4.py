import torch
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import BoxUniform
from matplotlib import pyplot as plt
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

num_rounds = 2
num_dim = 3
x_o = torch.zeros(num_dim,)

prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
simulator = lambda theta: theta + torch.randn_like(theta) * 0.1

prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)

density_estimator_build_fun = posterior_nn(
    model="nsf", hidden_features=60, num_transforms=3
)
inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
inference2 = NPE(prior=prior, density_estimator="maf")

posteriors = []
posteriors2 = []
proposal = prior
proposal2 = prior

for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    theta2, x2 = simulate_for_sbi(simulator, proposal2, num_simulations=500)

    # In `SNLE` and `SNRE`, you should not pass the `proposal` to
    # `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train()
    density_estimator2 = inference2.append_simulations(
        theta2, x2, proposal=proposal2
    ).train()
    posterior = inference.build_posterior(density_estimator)
    posterior2 = inference2.build_posterior(density_estimator2) 
    posteriors.append(posterior)
    posteriors2.append(posterior2)
    proposal = posterior.set_default_x(x_o)
    proposal2 = posterior2.set_default_x(x_o)

posterior_samples = posterior.sample((10000,), x=x_o)
posterior_samples2 = posterior2.sample((10000,), x=x_o)

# plot posterior samples
fig, ax = pairplot(
    posterior_samples, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5), title='NSF'
)
fig, ax = pairplot(
    posterior_samples2, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5), title='MAF'
)
plt.show()