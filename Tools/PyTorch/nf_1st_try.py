import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the Coupling Layer
class CouplingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2 * 2)  # output both scale and shift
        )

    def forward(self, x, mask):
        x1 = x * mask
        x2 = x * (1 - mask)
        
        # Flatten x1 based on mask
        x1_flattened = x1[:, mask.bool()]
        
        # Forward pass through net
        scale_shift = self.net(x1_flattened)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Debug: Print out some intermediate values
        print("x1_flattened mean:", x1_flattened.mean().item())
        print("scale mean before clamp:", scale.mean().item())
        print("shift mean:", shift.mean().item())

        # Apply sigmoid and clamp to prevent scale from being too large/small
        scale = torch.sigmoid(scale + 2.0)
        scale = torch.clamp(scale, min=1e-6, max=1.0)  # Clamp scale
        
        # Affine transformation
        y = x1 + x2 * scale + shift
        log_det_jacobian = torch.sum(torch.log(scale) * (1 - mask), dim=-1)
        
        # Debug: Check for NaNs in outputs
        assert not torch.isnan(y).any(), "NaN in output y"
        assert not torch.isnan(log_det_jacobian).any(), "NaN in log_det_jacobian"
        
        return y, log_det_jacobian

    def inverse(self, y, mask):
        y1 = y * mask
        y2 = y * (1 - mask)
        
        # Flatten y1 based on mask
        y1_flattened = y1[:, mask.bool()]
        
        # Forward pass through net
        scale_shift = self.net(y1_flattened)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Apply sigmoid and clamp to prevent scale from being too large/small
        scale = torch.sigmoid(scale + 2.0)
        scale = torch.clamp(scale, min=1e-6, max=1.0)  # Clamp scale

        # Inverse transformation
        x = y1 + (y2 - shift) / scale
        log_det_jacobian = -torch.sum(torch.log(scale) * (1 - mask), dim=-1)

        return x, log_det_jacobian
# Define the Normalizing Flow model
class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([CouplingLayer(input_dim) for _ in range(n_layers)])
        self.masks = [torch.tensor([1, 0] * (input_dim // 2), dtype=torch.float32) if i % 2 == 0
                      else torch.tensor([0, 1] * (input_dim // 2), dtype=torch.float32)
                      for i in range(n_layers)]
    
    def forward(self, x):
        log_det_jacobian = 0
        for layer, mask in zip(self.layers, self.masks):
            x, ldj = layer(x, mask)
            log_det_jacobian += ldj
        return x, log_det_jacobian
    
    def inverse(self, y):
        log_det_jacobian = 0
        for layer, mask in reversed(list(zip(self.layers, self.masks))):
            y, ldj = layer.inverse(y, mask)
            log_det_jacobian += ldj
        return y, log_det_jacobian

# Define the target distribution (a mixture of Gaussians)
def target_distribution(x):
    return 0.5 * (torch.exp(-0.5 * ((x - 2) / 0.8)**2) + torch.exp(-0.5 * ((x + 2) / 0.8)**2))

# Training the Normalizing Flow
input_dim = 2
flow = NormalizingFlow(input_dim)
optimizer = optim.Adam(flow.parameters(), lr=1e-3)
base_dist = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

num_samples = 1000
num_epochs = 1000

for epoch in range(num_epochs):
    # Sample from the base distribution
    z = base_dist.sample((num_samples,))
    
    # Map to the target distribution using the normalizing flow
    x, log_det_jacobian = flow.inverse(z)
    
    # Compute the target log probability, add small constant to avoid log(0)
    log_prob = torch.log(target_distribution(x) + 1e-6).sum(dim=-1)
    
    # Compute the flow loss
    loss = -(log_prob + log_det_jacobian).mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate and check samples
with torch.no_grad():
    z = base_dist.sample((num_samples,))
    samples, _ = flow.inverse(z)
    
    # Check for NaN values
    assert not torch.isnan(samples).any(), "Samples contain NaN values."
    
    # Print sample statistics for diagnostics
    print("Sample stats - mean:", samples.mean(dim=0))
    print("Sample stats - std:", samples.std(dim=0))
    print("Sample min:", samples.min(dim=0)[0])
    print("Sample max:", samples.max(dim=0)[0])

    samples = samples.numpy()

plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.title("Samples from the target distribution using normalizing flow")
plt.show()
