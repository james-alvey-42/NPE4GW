import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as D


samples = []
N=20000
def p(x):
    return x**2*np.exp(-3*x)

x_current = 1
for i in range(N):
    x_proposed = np.random.normal(x_current, 0.1)
    if x_proposed > 0:
        r = p(x_proposed)/p(x_current)
    else:
        r = 0
    u = np.random.uniform(0,1)
    if u <= min(1,r):
        x_current = x_proposed
    samples.append(x_current)
    
#return final 5000 samples
samples = samples[5000:]

plt.hist(samples, bins=100, density=True)
x = torch.linspace(0,3,100)
# plot Gamma(3,3) for comparison
plt.plot(x, D.Gamma(3,3).log_prob(x).exp().numpy())
plt.legend(['True Distribution', 'MCMC Samples'])
plt.title('Metropolis-Hastings Samples')
plt.show()
