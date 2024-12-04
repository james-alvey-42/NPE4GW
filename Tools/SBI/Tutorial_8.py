import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from sbi.inference import NPE
from sbi.analysis import pairplot
from sbi.utils import BoxUniform

prior_min = [-1, -1, -1]
prior_max = [1, 1, 1]
prior = BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

def create_t_x(theta, seed=None):
    """Return an t, x array for plotting based on params"""
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    t = np.linspace(-1, 1, 200)
    ts = np.repeat(t[:, np.newaxis], theta.shape[0], axis=1)
    x = (
        theta[:, 0] * ts**2
        + theta[:, 1] * ts
        + theta[:, 2]
        + 0.01 * rng.randn(ts.shape[0], theta.shape[0])
    )
    return t, x

def eval(theta, t, seed=None):
    """Evaluate the quadratic function at `t`"""

    if theta.ndim == 1:
        theta = theta[np.newaxis, :]

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    return theta[:, 0] * t**2 + theta[:, 1] * t + theta[:, 2] + 0.01 * rng.randn(1)

def get_3_values(theta, seed=None):
    """
    Return 3 'x' values corresponding to t=-0.5,0,0.75 as summary statistic vector
    """
    return np.array(
        [
            eval(theta, -0.5, seed=seed),
            eval(theta, 0, seed=seed),
            eval(theta, 0.75, seed=seed),
        ]
    ).T

def get_MSE(theta, theta_o, seed=None):
    """
    Return the mean-squared error (MSE) i.e. Euclidean distance from the
    observation function
    """
    _, x = create_t_x(theta_o, seed=seed)  # truth
    _, x_ = create_t_x(theta, seed=seed)  # simulations
    return np.mean(np.square(x_ - x), axis=0, keepdims=True).T  # MSE

theta_o = np.array([0.3, -0.2, -0.1])
t, x_truth = create_t_x(theta_o)
plt.plot(t, x_truth, "k", zorder=1, label="truth")
n_samples = 100
theta = prior.sample((n_samples,))
t, x = create_t_x(theta.numpy())
# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.plot(t, x, "grey", zorder=0)
plt.legend()
plt.title("Prior Samples vs Truth")

theta = prior.sample((1000,))
x = get_MSE(theta.numpy(), theta_o)
theta = torch.as_tensor(theta, dtype=torch.float32)
x = torch.as_tensor(x, dtype=torch.float32)

inference = NPE(prior)
_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

x_o = torch.as_tensor(
    [
        [
            0.0,
        ]
    ]
)
theta_p = posterior.sample((10000,), x=x_o)

fig, axes = pairplot(
    theta_p,
    limits=list(zip(prior_min, prior_max)),
    ticks=list(zip(prior_min, prior_max)),
    figsize=(7, 7),
    labels=["a", "b", "c"],
    fig_kwargs=dict(
        points_offdiag={"markersize": 6},
        points_colors="r",
    ),
    points=theta_o,
    title="Posterior samples - MSE summary statistic",
);

theta_p = posterior.sample((100,), x=x_o)
x_t, x_x = create_t_x(theta_p.numpy())
plt.plot(x_t, x_x, "grey", zorder=0)
plt.legend();

theta1 = prior.sample((1000,))
theta1 = torch.as_tensor(theta1, dtype=torch.float32)
x1 = get_3_values(theta1.numpy())
x1= torch.as_tensor(x1, dtype=torch.float32)

inference1 = NPE(prior)

_ = inference1.append_simulations(theta1, x1).train()
posterior1 = inference1.build_posterior()

x_o = torch.as_tensor(get_3_values(theta_o), dtype=float)

theta_p1 = posterior1.sample((10000,), x=x_o)

fig, axes = pairplot(
    theta_p1,
    limits=list(zip(prior_min, prior_max)),
    ticks=list(zip(prior_min, prior_max)),
    figsize=(7, 7),
    labels=["a", "b", "c"],
    fig_kwargs=dict(
        points_offdiag={"markersize": 6},
        points_colors="r",
    ),
    points=theta_o,
    title= "Posterior samples - 3 values summary statistic",
);

x_o_t, x_o_x = create_t_x(theta_o)
plt.plot(x_o_t, x_o_x, "k", zorder=1, label="truth")
theta_p1 = posterior1.sample((100,), x=x_o)
ind_10_highest = np.argsort(np.array(posterior1.log_prob(theta=theta_p1, x=x_o)))[-10:]
theta_p_considered = theta_p1[ind_10_highest, :]
x_t, x_x = create_t_x(theta_p_considered.numpy())
plt.plot(x_t, x_x, "grey", zorder=0)
plt.legend()

stats = np.concatenate(
    (get_3_values(theta.numpy()), get_MSE(theta.numpy(), theta_o)), axis=1
)
x_o = np.concatenate((get_3_values(theta_o), np.asarray([[0.0]])), axis=1)

features = ["x @ t=-0.5", "x @ t=0", "x @ t=0.7", "MSE"]
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
xlabelfontsize = 10
for i, ax in enumerate(axes.reshape(-1)):
    ax.hist(
        stats[:, i],
        color=["grey"],
        alpha=0.5,
        bins=30,
        density=True,
        histtype="stepfilled",
        label=["simulations"],
    )
    ax.axvline(x_o[:, i], label="observation", color='k')
    ax.set_xlabel(features[i], fontsize=xlabelfontsize)
    if i == 3:
        ax.legend()
plt.tight_layout()

plt.show()

