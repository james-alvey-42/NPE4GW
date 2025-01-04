import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn

class CustomEmbeddingNet5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super(CustomEmbeddingNet5, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, batch):
        """
        Forward pass for the embedding network, accepting a Batch object.

        Args:
            batch: PyTorch Geometric Batch object containing `x`, `edge_index`, and `batch`.
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        # Pass through GCN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Global mean pooling
        graph_embeddings = global_mean_pool(x, batch_index)

        # Pass through fully connected layers
        return self.fc(graph_embeddings)

# Define the simulator
def simulator(theta):
    """
    Simulates graph data where node features depend on parameters theta.
    
    Args:
        theta: Tensor of shape [batch_size, num_params].
    Returns:
        List of PyTorch Geometric Data objects (graphs).
    """
    batch_size = theta.shape[0]
    num_nodes = 10  # Fixed number of nodes
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                               dtype=torch.long).t()  # Fully connected graph
    
    graphs = []
    for i in range(batch_size):
        node_features = theta[i, 0].item() * torch.rand((num_nodes, 1)) + theta[i, 1].item()
        graphs.append(Data(x=node_features, edge_index=edge_index))
    return graphs

# Define prior and model
prior = BoxUniform(low=torch.tensor([0.1, -1.0]), high=torch.tensor([1.0, 1.0]))
input_dim = 1       # Node feature dimension
hidden_dim = 32     # GNN hidden dimension
output_dim = 2      # Parameter space dimension
num_nodes = 10      # Number of nodes in each graph
embedding_net = CustomEmbeddingNet5(input_dim, hidden_dim, output_dim, num_nodes)

# Instantiate the posterior network using MAF
density_estimator = posterior_nn(model='maf', embedding_net=embedding_net)

# Instantiate SNPE
inference = SNPE(prior=prior, density_estimator=density_estimator)

# Simulate data
theta_samples = prior.sample((1024,))  # Sample 1024 parameter sets
graphs = simulator(theta_samples)

# Use DataLoader to batch graphs
graph_loader = DataLoader(graphs, batch_size=32, shuffle=True)

# Process batches through the embedding network
x_embeddings = []
for batch in graph_loader:
    # Ensure the batch is a Batch (DataBatch is a subclass of Batch)
    if isinstance(batch, Batch):
        # Unpack the necessary components
        x = batch.x  # Node features: [batch_size * num_nodes, input_dim]
        edge_index = batch.edge_index  # Edge indices: [2, num_edges]
        batch_index = batch.batch  # Batch indices: [batch_size * num_nodes]

        # Pass through the embedding network
        output = embedding_net(x, batch_index)  # Ensure this correctly passes the inputs to your model
        x_embeddings.append(output)

# Concatenate embeddings from all batches
x = torch.cat(x_embeddings, dim=0)


# Concatenate embeddings from all batches
x = torch.cat(x_embeddings, dim=0)

# Append simulations and train SNPE
inference.append_simulations(theta_samples, x).train()

# Build the posterior
posterior = inference.build_posterior(density_estimator)

# Simulate a new observation (single graph or small batch)
theta_test = prior.sample((1,))
graph_test = simulator(theta_test)
graph_batch_test = Batch.from_data_list(graph_test)  # Convert list of graphs into a Batch

# Embed the test graph
x_test = embedding_net(graph_batch_test.x, graph_batch_test.batch)

# Sample posterior given the test observation
posterior_samples = posterior.sample((1000,), x=x_test)

print("True theta:", theta_test)
print("Posterior mean:", posterior_samples.mean(dim=0))
