import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        # Register the mask as a buffer to avoid parameterization
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Debugging: Check the shapes
        print(f"Weight shape: {self.weight.shape}, Mask shape: {self.mask.shape}")
        assert self.weight.shape == self.mask.shape, "Weight and mask must have the same shape!"
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(MADE, self).__init__()
        self.input_dim = input_dim

        # Generate masks
        self.masks = self.create_masks(input_dim, hidden_dim, num_hidden_layers)

        # Build network layers
        layers = []
        in_dim = input_dim
        for i in range(num_hidden_layers):
            out_dim = hidden_dim
            layers.append(MaskedLinear(in_dim, out_dim, self.masks[i]))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final output layer
        layers.append(MaskedLinear(in_dim, input_dim, self.masks[-1]))
        self.net = nn.Sequential(*layers)

    def create_masks(self, input_dim, hidden_dim, num_hidden_layers):
        """Generate masks for each layer."""
        masks = []

        # Degrees for inputs and hidden neurons
        input_degrees = torch.arange(1, input_dim + 1)
        hidden_degrees = [
            torch.randint(1, input_dim + 1, (hidden_dim,))
            for _ in range(num_hidden_layers)
        ]

        # Mask: input to first hidden layer
        masks.append((hidden_degrees[0][:, None] >= input_degrees[None, :]).float())

        # Masks: hidden to hidden layers
        for i in range(1, num_hidden_layers):
            masks.append((hidden_degrees[i][:, None] >= hidden_degrees[i - 1][None, :]).float())

        # Mask: last hidden to output layer
        masks.append((input_degrees[None, :] >= hidden_degrees[-1][:, None]).float())

        # Debugging: Print all mask shapes
        for i, mask in enumerate(masks):
            print(f"Mask {i} shape: {mask.shape}")

        return masks

    def forward(self, x):
        return self.net(x)


class MAF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_flows):
        super(MAF, self).__init__()
        self.flows = nn.ModuleList([
            MADE(input_dim, hidden_dim, num_hidden_layers)
            for _ in range(num_flows)
        ])
        self.base_dist = torch.distributions.Normal(0, 1)

    def forward(self, x):
        log_det_jacobian = 0
        for flow in self.flows:
            z = flow(x)
            x = z
        return z, log_det_jacobian

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        log_base_prob = torch.sum(self.base_dist.log_prob(z), dim=1)
        return log_base_prob + log_det_jacobian


# Example usage
if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 32
    num_hidden_layers = 2
    num_flows = 3

    maf = MAF(input_dim, hidden_dim, num_hidden_layers, num_flows)

    x = torch.randn(10, input_dim)
    log_prob = maf.log_prob(x)
    print("Log probabilities:", log_prob)
