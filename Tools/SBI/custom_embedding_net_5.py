import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as sbi_utils
import matplotlib.pyplot as plt


# Define GNN-based embedding network
class CustomEmbeddingNetGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomEmbeddingNetGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, batch_data):
        x, edge_index = batch_data.x, batch_data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # Pooling: mean over all nodes in each graph
        batch = batch_data.batch
        return torch.cat([x[batch == i].mean(dim=0, keepdim=True) for i in torch.unique(batch)], dim=0)


# Simulator function to generate graph-based data
def simulator(theta):
    """
    Simulates graph data based on input parameters theta.
    Args:
        theta: Torch tensor of shape [num_samples, 3].
    Returns:
        List of torch_geometric.data.Data objects (one per sample).
    """
    num_samples = theta.shape[0]
    num_nodes = 10
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()  # Fully connected graph

    graphs = []
    for i in range(num_samples):
        # Node features based on theta
        node_features = torch.stack([
            theta[i, 0] * torch.ones(num_nodes),  # Feature 1 (based on mean_x)
            theta[i, 1] * torch.linspace(0, 1, num_nodes),  # Feature 2 (based on mean_y)
            theta[i, 2] * torch.randn(num_nodes)  # Feature 3 (random noise based on mean_z)
        ], dim=1)
        graphs.append(Data(x=node_features, edge_index=edge_index))

    return graphs


# Prior over the parameters
prior = sbi_utils.BoxUniform(low=torch.tensor([-1.0, 1.0, 0.1]), 
                             high=torch.tensor([1.0, 10.0, 2.0]))

# Instantiate the embedding network
input_dim = 3       # Node feature dimension
hidden_dim = 16     # Hidden layer size in GNN
output_dim = 3      # Parameter dimension
embedding_net = CustomEmbeddingNetGNN(input_dim, hidden_dim, output_dim)

density_estimator = posterior_nn(model="maf", embedding_net=embedding_net)
inference = SNPE(prior=prior, density_estimator=density_estimator)

# Generate simulated data
num_simulations = 500
theta = prior.sample((num_simulations,))
graph_data_list = simulator(theta)  # List of graphs
batched_graphs = Batch.from_data_list(graph_data_list)  # Batch all graphs

# Flatten batched graph node features and pass to SNPE
x = embedding_net(batched_graphs)

# Train the density estimator
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)

# Define a test parameter set
theta_test = torch.tensor([0.5, 7.5, 0.8])  # Test parameter
test_graph = simulator(theta_test.unsqueeze(0))[0]  # Simulate data for this parameter
x_test = embedding_net(Batch.from_data_list([test_graph]))  # Process test graph through embedding network

# Sample from the posterior
samples = posterior.sample((1000,), x=x_test)

# Visualize marginal posterior distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
params = ["mean_x", "mean_y", "mean_z"]
for i in range(3):
    axes[i].hist(samples[:, i].numpy(), bins=30, alpha=0.7, color="blue", density=True)
    axes[i].axvline(theta_test[i].item(), color="red", linestyle="--", label="True")
    axes[i].set_title(f"Posterior of {params[i]}")
    axes[i].legend()

plt.tight_layout()
plt.show()
