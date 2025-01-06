import numpy as np
from scipy import stats
import pylab as plt
import torch
import swyft
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
np.random.seed(0)

class Simulator(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.x = np.linspace(-1, 1, 10)
         
    def build(self, graph):
        z = graph.node('z', lambda: np.random.rand(2)*2-1)
        f = graph.node('f', lambda z: z[0] + z[1]*self.x, z)
        x = graph.node('x', lambda f: f + np.random.randn(10)*0.1, f)

class Network(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(10, 2)
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 2, num_params = 2, varnames = 'z')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 2, marginals = ((0, 1),), varnames = 'z')

    def forward(self, A, B):
        embedding = self.embedding(A['x'])
        logratios1 = self.logratios1(embedding, B['z'])
        logratios2 = self.logratios2(embedding, B['z'])
        return logratios1, logratios2

sim = Simulator()
samples = sim.sample(N = 10000)

trainer = swyft.SwyftTrainer(accelerator = DEVICE)
dm = swyft.SwyftDataModule(samples, batch_size = 64)
network = Network()
trainer.fit(network, dm)

test_samples = sim.sample(N = 1000)
trainer.test(network, test_samples.get_dataloader(batch_size = 64))
B = samples[:1000]
A = samples[:1000]
mass = trainer.test_coverage(network, A, B)

z0 = np.array([0.3, 0.7])
x0 = sim.sample(conditions = {"z": z0})['x']
plt.plot(x0)

prior_samples = sim.sample(targets = ['z'], N = 100000)

predictions = trainer.infer(network, swyft.Sample(x = x0), prior_samples)

predictions[0].parnames

swyft.plot_posterior(predictions, ['z[0]', 'z[1]']);
plt.show()