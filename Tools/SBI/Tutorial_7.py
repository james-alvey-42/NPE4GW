import torch
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform, RestrictionEstimator
_ = torch.manual_seed(2)
from sbi.inference import NPE
from sbi.utils import RestrictionEstimator
from matplotlib import pyplot as plt

def simulator(theta):
    perturbed_theta = theta + 0.5 * torch.randn(2)
    perturbed_theta[theta[:, 0] < 0.0] = torch.as_tensor([float("nan"), float("nan")])
    return perturbed_theta

num_rounds = 5

prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
proposals = [prior]

theta, x = simulate_for_sbi(simulator, prior, 1000)
print("Simulation outputs: ", x)

restriction_estimator = RestrictionEstimator(prior=prior)

for r in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposals[-1], 1000)
    restriction_estimator.append_simulations(theta, x)
    if (
        r < num_rounds - 1
    ):  # training not needed in last round because classifier will not be used anymore.
        classifier = restriction_estimator.train()
    proposals.append(restriction_estimator.restrict_prior())

restricted_prior = restriction_estimator.restrict_prior()
samples = restricted_prior.sample((10_000,))
_ = pairplot(samples, limits=[[-2, 2], [-2, 2]], figsize=(4, 4), title="Restricted prior")

new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, 1000)
print("Simulation outputs: ", new_x)
# Print how many NaNs are in the simulated data
num_nans = torch.isnan(new_x).sum().item()
print(f"Number of NaNs in the simulated data: {num_nans}")

restriction_estimator.append_simulations(
    new_theta, new_x
)  # Gather the new simulations in the `restriction_estimator`.
all_theta, all_x, _ = restriction_estimator.get_simulations()  # Get all simulations run so far.

inference = NPE(prior=prior)
density_estimator = inference.append_simulations(all_theta, all_x).train()
#print how many NaNs are in the data
posterior = inference.build_posterior()

posterior_samples = posterior.sample((10_000,), x=torch.ones(2))
_ = pairplot(posterior_samples, limits=[[-2, 2], [-2, 2]], figsize=(3, 3), title="Posterior samples")
plt.show()