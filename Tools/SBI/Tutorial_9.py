from torch import ones, eye
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

from sbi.inference import NPE, ImportanceSamplingPosterior
from sbi.utils import BoxUniform
from sbi.inference.potentials.base_potential import BasePotential
from sbi.analysis import marginal_plot

# define prior and simulator
class Simulator:
    def __init__(self):
        pass

    def log_likelihood(self, theta, x):
        return MultivariateNormal(theta, eye(2)).log_prob(x)

    def sample(self, theta):
        return theta + torch.randn((theta.shape))

prior = BoxUniform(-5 * ones((2,)), 5 * ones((2,)))
sim = Simulator()
log_prob_fn = lambda theta, x_o: sim.log_likelihood(theta, x_o) + prior.log_prob(theta)

# generate train data
_ = torch.manual_seed(3)
theta = prior.sample((10,))
x = sim.sample(theta)

# train NPE model
_ = torch.manual_seed(4)
inference = NPE(prior=prior)
_ = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior()

# generate a synthetic observation
_ = torch.manual_seed(2)
theta_gt = prior.sample((1,))
observation = sim.sample(theta_gt)[0]
posterior = posterior.set_default_x(observation)
print("observations.shape", observation.shape)

# sample from posterior
theta_inferred = posterior.sample((10_000,))

# get samples from ground-truth posterior
gt_samples = MultivariateNormal(observation, eye(2)).sample((len(theta_inferred) * 5,))
gt_samples = gt_samples[prior.support.check(gt_samples)][:len(theta_inferred)]

posterior_sir = ImportanceSamplingPosterior(
    potential_fn=log_prob_fn,
    proposal=posterior,
    method="sir",
).set_default_x(observation)

theta_inferred_sir = posterior_sir.sample(
    (1000,),
    oversampling_factor=32,
)

fig, ax = marginal_plot(
    [theta_inferred, theta_inferred_sir, gt_samples],
    limits=[[-5, 5], [-5, 5]],
    figsize=(5, 1.5),
    diag="kde",  # smooth histogram
)
ax[1].legend(["NPE", "NPE-IS", "Groud Truth"], loc="upper right", bbox_to_anchor=[2.0, 1.0, 0.0, 0.0]);

plt.show()
