import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib import animation, rc

from sbi.analysis import (
    conditional_corrcoeff,
    conditional_pairplot,
    conditional_potential,
    pairplot,
)

_ = torch.manual_seed(0)
from toy_posterior_for_07_cc import ExamplePosterior

posterior = ExamplePosterior()

x_o = torch.ones(1, 20)  # simulator output was 20-dimensional
posterior.set_default_x(x_o)

posterior_samples = posterior.sample((5000,))

fig, ax = pairplot(
    samples=posterior_samples,
    limits=torch.tensor([[-2.0, 2.0]] * 3),
    upper=["kde"],
    diag=["kde"],
    figsize=(5, 5),
)

corr_matrix_marginal = np.corrcoef(posterior_samples.T)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = plt.imshow(corr_matrix_marginal, clim=[-1, 1], cmap="PiYG")
_ = fig.colorbar(im)

rc("animation", html="html5")

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))


def init():
    (line,) = ax.plot([], [], lw=2)
    line.set_data([], [])
    return (line,)


def animate(angle):
    num_samples_vis = 1000
    line = ax.scatter(
        posterior_samples[:num_samples_vis, 0],
        posterior_samples[:num_samples_vis, 1],
        posterior_samples[:num_samples_vis, 2],
        zdir="z",
        s=15,
        c="#2171b5",
        depthshade=False,
    )
    ax.view_init(20, angle)
    return (line,)


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=range(0, 360, 5), interval=150, blit=True
)

plt.close()

plt.show()
