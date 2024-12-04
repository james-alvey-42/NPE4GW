import matplotlib.pyplot as plt
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
)
from sbi.utils.metrics import c2st
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.neural_nets import posterior_nn

# Seeding
torch.manual_seed(1);

# Gaussian simulator
theta_dim = 2
x_dim = theta_dim

# likelihood_mean will be likelihood_shift+theta
likelihood_shift = -1.0 * zeros(x_dim)
likelihood_cov = 0.3 * eye(x_dim)

prior_mean = zeros(theta_dim)
prior_cov = eye(theta_dim)
prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

# Define Gaussian simulator
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(
    lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov),
    prior,
    prior_returns_numpy,
)
check_sbi_inputs(simulator, prior) 

max_num_trials = 20

# construct training data set: we want to cover the full range of possible number of
# trials
num_training_samples = 1000
theta = prior.sample((num_training_samples,))

# there are certainly smarter ways to construct the training data set, but we go with a
# for loop here for illustration purposes.
x = torch.ones(num_training_samples * max_num_trials, max_num_trials, x_dim) * float(
    "nan"
)
for i in range(num_training_samples):
    xi = simulator(theta[i].repeat(max_num_trials, 1))
    for j in range(max_num_trials):
        x[i * max_num_trials + j, : j + 1, :] = xi[: j + 1, :]

theta = theta.repeat_interleave(max_num_trials, dim=0)

# embedding
latent_dim = 10
single_trial_net = FCEmbedding(
    input_dim=theta_dim,
    num_hiddens=40,
    num_layers=2,
    output_dim=latent_dim,
)
embedding_net = PermutationInvariantEmbedding(
    single_trial_net,
    trial_net_output_dim=latent_dim,
    # NOTE: post-embedding is not needed really.
    num_layers=1,
    num_hiddens=10,
    output_dim=10,
)

# we choose a simple MDN as the density estimator.
# NOTE: we turn off z-scoring of the data, as we used NaNs for the missing trials.
density_estimator = posterior_nn("maf", embedding_net=embedding_net, z_score_x="none")

inference = NPE(prior, density_estimator=density_estimator)
# NOTE: we don't exclude invalid x because we used NaNs for the missing trials.
inference.append_simulations(
    theta,
    x,
    exclude_invalid_x=False,
).train(training_batch_size=1000)
posterior = inference.build_posterior()

num_trials = [1, 5, 15, 20]
theta_o = zeros(1, theta_dim)
xos = [theta_o.repeat(nt, 1) for nt in num_trials]

npe_samples = []
for xo in xos:
    # we need to pad the x_os with NaNs to match the shape of the training data.
    xoi = torch.ones(1, max_num_trials, x_dim) * float("nan")
    xoi[0, : len(xo), :] = xo
    npe_samples.append(posterior.sample(sample_shape=(num_training_samples,), x=xoi))


# Plot them in one pairplot as contours (obtained via KDE on the samples).
fig, ax = pairplot(
    npe_samples,
    points=theta_o,
    diag="kde",
    upper="contour",
    diag_kwargs=dict(bins=100),
    upper_kwargs=dict(levels=[0.95]),
    fig_kwargs=dict(
        points_colors=["k"],
        points_offdiag=dict(marker="*", markersize=10
        ),
    )
)
plt.sca(ax[1, 1])
plt.legend(
    [f"{nt} trials" if nt > 1 else f"{nt} trial" for nt in num_trials]
    + [r"$\theta_o$"],
    frameon=False,
    fontsize=12,
);
plt.show()