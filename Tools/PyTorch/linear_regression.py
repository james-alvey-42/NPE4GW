#dependencies 
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger


class LinearRegressionModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearRegressionModel, self).__init__()
        # Define the linear layer
        # Initialize weights and biases - nn.Parameter means they're trainable
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))  # Weight matrix of shape (output_dim, input_dim)
        self.bias = nn.Parameter(torch.randn(output_dim))  # Bias vector of shape (output_dim)
    
    def forward(self, x):
        # Forward pass
        return x @ self.weight.t() + self.bias
    
    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Calculate loss - MSE chosen here
        N = len(y_hat)
        e = torch.norm(y_hat-y)
        E = torch.square(e)
        loss = 1/N * E
        #log loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
    # Unpack the batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Calculate loss - MSE chosen here
        N = len(y_hat)
        e = torch.norm(y_hat-y)
        E = torch.square(e)
        val_loss = 1/N * E
        #log loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        # Set up an optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
    
    # def on_epoch_end(self):
    #     # Make predictions on validation data
    #     self.eval()
    #     with torch.no_grad():
    #         y_pred = self(self.x_range)
    #         self.predictions_over_time.append(y_pred.cpu().numpy())  # Store predictions
    #     self.train()
# Generate data
torch.manual_seed(42)  # For reproducibility
x = torch.randn(1000, 1)  # 1000 samples, 1 feature
y = 5 * x + 2 + 0.4 * torch.randn(1000, 1)  # Linear relation with some noise
# Create DataLoader and split into Validation and Training sets
dataset = TensorDataset(x, y)
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)
# Initialize model
model = LinearRegressionModel(input_dim=1, output_dim=1)
#Set up the Logger
logger = TensorBoardLogger("tb_logs", name="linear_regression")
# Set up trainer
trainer = pl.Trainer(max_epochs=10, logger=logger)
# Train model
trainer.fit(model, train_loader, valid_loader)
# Switch model to evaluation mode
model.eval()
# Use the model to predict y values for the input x
with torch.no_grad():
    predicted_y = model(x)
# Convert tensors to numpy arrays for plotting
x_np = x.numpy()
y_np = y.numpy()
predicted_y_np = predicted_y.numpy()
# Plot the data and the model's predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_np, y_np, label='Actual Data', color='blue', alpha=0.6)
plt.plot(x_np, predicted_y_np, label='Model Prediction', color='red', linewidth=2)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Linear Regression Model Fit')
plt.legend()
plt.show()
