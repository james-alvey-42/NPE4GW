import torch
import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
num_dim = 3

def simulator(theta):
    # linear gaussian
    return theta + 1.0 + torch.randn_like(theta) * 0.1

prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

inference = NPE(prior=prior)

num_simulations = 2000
theta = prior.sample((num_simulations,))
x = simulator(theta)
print("theta.shape", theta.shape)
print("x.shape", x.shape)

inference = inference.append_simulations(theta, x)

density_estimator = inference.train()

posterior = inference.build_posterior(density_estimator)

print(posterior) # prints how the posterior was trained

theta_true = prior.sample((1,))
# generate our observation
x_obs = simulator(theta_true)

theta_posterior = posterior.sample((10000,), x=x_obs)  # sample from posterior
x_predictive = simulator(theta_posterior)  # simulate data from posterior
pairplot(x_predictive,
         points=x_obs,  # plot with x_obs as a point
         figsize=(6, 6),
         labels=[r"$x_1$", r"$x_2$", r"$x_3$"]);
#show the plot of the posterior

# first sample an alternative parameter set from the prior
theta_diff = prior.sample((1,))

log_probability_true_theta = posterior.log_prob(theta_true, x=x_obs)
log_probability_diff_theta = posterior.log_prob(theta_diff, x=x_obs)
log_probability_samples = posterior.log_prob(theta_posterior, x=x_obs)

print( r'high for true theta :', log_probability_true_theta)
print( r'low for different theta :', log_probability_diff_theta)
print( r'range of posterior samples: min:', torch.min(log_probability_samples),' max :', torch.max(log_probability_samples))

plt.show()
