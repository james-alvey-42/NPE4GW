### IMPORTS ###
import sbi
import torch
from sbi.neural_nets.net_builders import build_nsf
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior, ImportanceSamplingPosterior
import matplotlib.pyplot as plt
import corner
from sbi.utils import BoxUniform
import wandb
import tqdm
from sbi.analysis import pairplot

### EXAMPLE SIMULATOR ###

def simulator(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1

### EXAMPLE EMBEDDING NET ###

class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
        )

    def forward(self, x):
        return self.net(x)

### CUSTOM DATASET ### 
# Consider creating your own here or can use the one implemented in swyft
# Useful to learn how to write custom dataloaders anyway, they are v useful

class NPEData(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int, prior: torch.distributions.Distribution, simulator
    ):
        super().__init__()
        self.prior = prior
        self.simulator = simulator

        self.theta = prior.sample((num_samples,))
        self.x = simulator(self.theta)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int):
        return self.theta[index, ...], self.x[index, ...]

### HELPER FUNCTION TO CREATE DENSITY ESIMATOR ###

def build_density_estimator(prior, simulator):
    embedding_net = EmbeddingNet(in_features=3)
    dummy_data = NPEData(num_samples=64, prior=prior, simulator=simulator)
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        embedding_net=embedding_net,
    )
    return density_estimator

### EXAMPLE LR SCHEDULER ###
# Again, want to experiment here, lots of options

def setup_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    )
    return scheduler


if __name__ == "__main__":
    wandb.init(project="npe4gw") # initialise wandb

    prior = BoxUniform(low=-2 * torch.ones(3), high=2 * torch.ones(3)) # define a prior

    train_data = NPEData(num_samples=10_000, prior=prior, simulator=simulator) # setup train dataset
    val_data = NPEData(num_samples=1_000, prior=prior, simulator=simulator) # setup val dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    ) # Create train dataloader from dataset
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True) # Create val dataloader from dataset

    density_estimator = build_density_estimator(prior, simulator) # initialise density estimator
    optimizer = AdamW(density_estimator.parameters(), lr=1e-3) # initialise pytorch optimiser
    scheduler = setup_scheduler(optimizer) # initialise scheduler
    train_losses = [] # setup for loss tracking (in addition to wandb)
    val_losses = [] # setup for loss tracking (in addition to wandb)
    learning_rates = [] # setup for loss tracking (in addition to wandb)
    post_samples = [] # setup for loss tracking (in addition to wandb)

    num_epochs = 10
    step = 0
    epoch_val_loss = 0.0
    
    ### CUSTOM TRAINING LOOP ###

    for epoch in range(num_epochs):
        density_estimator.train() # put estimator into train mode

        train_loss_epoch = 0.0
        num_batches = len(train_dataloader)
        with tqdm.tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ) as pbar: # Fancy tqdm loading bar for printing the training status
            for theta, x in pbar: # Iterate through the dataloader batches (called pbar here)
                loss = density_estimator.loss(theta, x).mean() # compute loss on batch
                optimizer.zero_grad() # zero the optimiser
                loss.backward() # compute the gradients
                optimizer.step() # take a step given these gradients
                train_losses.append(loss.item()) # track losses
                train_loss_epoch += loss.item()
                wandb.log({"train_loss": loss.item(), "step": step}) # log loss to wandb
                step += 1
                pbar.set_postfix(
                    {
                        "Train Loss": f"{loss.item():.4f} | Val Loss: {epoch_val_loss:.4f}"
                    }
                ) # print to tqdm bar

        avg_train_loss = train_loss_epoch / num_batches

        density_estimator.eval() # put estimator into eval mode

        epoch_val_loss = 0.0
        with torch.no_grad(): # ensure no gradients computed in val mode
            for theta, x in val_dataloader: # iterate through validation dataloader
                epoch_val_loss += density_estimator.loss(theta, x).mean().item() # compute overall loss on val dataset batch by batch

        epoch_val_loss /= len(val_dataloader) # average loss over val dataset
        val_losses.append(epoch_val_loss) # track loss
        scheduler.step(epoch_val_loss) # modify learning rate if necessary according to val loss metric
        learning_rates.append(scheduler.get_last_lr()) # track LR
        wandb.log(
            {
                "val_loss": epoch_val_loss,
                "step": step,
                "learning_rate": scheduler.get_last_lr(),
            }
        ) # log results

    x_o = torch.as_tensor([[1.0, 1.0, 1.0]])
    posterior = DirectPosterior(density_estimator, prior)
    samples = posterior.sample((1000,), x=x_o)
    _ = pairplot(samples, limits=[[-3, 3], [-3, 3], [-3, 3]], figsize=(3, 3), upper="contour")
    plt.show()

