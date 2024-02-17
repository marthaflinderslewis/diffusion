import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from itertools import islice

import copy
import random
import time

SEED = 1729

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

rng = np.random.default_rng()

ROOT = '.data'

train_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True)

mean = train_data.data.float().mean() / 255
std = train_data.data.float().std() / 255

train_transforms = transforms.Compose([
                            # transforms.RandomRotation(5, fill=(0,)),
                            # transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[mean], std=[std])
                                      ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[mean], std=[std])
                                     ])

train_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True,
                            transform=train_transforms)

test_data = datasets.MNIST(root=ROOT,
                           train=False,
                           download=True,
                           transform=test_transforms)

BATCH_SIZE = 4096

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

(x, y) = next(iter(train_iterator))
print(x.shape, y.shape)

def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='bone')
        ax.axis('off')
    plt.show()


N_IMAGES = 25

images = [image for image, label in [train_data[i] for i in range(N_IMAGES)]]

# plot_images(images)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 1000)
        self.hidden_fc = nn.Linear(1000, 1000)
        self.hidden_fc2 = nn.Linear(1000, 1000)
        self.output_fc = nn.Linear(1000, output_dim)
        self.double()

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 1000]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 1000]

        h_3 = F.relu(self.hidden_fc(h_2))
        # h_2 = [batch size, 1000]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2

INPUT_DIM = 28 * 28
OUTPUT_DIM = 28 * 28

model = MLP(INPUT_DIM, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters())

criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is: {device}', flush=True)

model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion, device, T=500, beta_start=1e-4, beta_end=1e-1):
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, (x0, _) in enumerate(iterator):#tqdm(iterator, desc="Training", leave=False):
        if i%10==0:
            print(f'done {i} of {len(iterator)}', flush=True)
        x0 = x0.to(device)
        bsz = x0.shape[0]
        x0=x0.reshape([bsz, -1])
        # choose a timestep
        t = random.randint(0, T-1)
        # create some noise
        y = rng.multivariate_normal(np.zeros(28*28), np.eye(28*28), bsz)
        y.reshape([bsz, -1])
        y = torch.from_numpy(y)
        y = y.to(device)

        # generate alpha_t
        alpha_t = (1-betas[t])**T

        # generate x_t
        x_t = x0*(alpha_t**0.5) + y*((1-alpha_t)**0.5)
        # x_t = x_t.to(torch.float32)


        optimizer.zero_grad()

        y_pred, _ = model(x_t)

        loss = criterion(y_pred, y)


        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def sample_and_plot(model, betas, T=10):
    fig, ax = plt.subplots()
    artists = []
    with torch.no_grad():
        x_current = rng.multivariate_normal(np.zeros(28*28), np.eye(28*28))
        x_current = torch.from_numpy(x_current)
        x_current = x_current.unsqueeze(0)
        print(x_current.shape)
        for t in range(T-1, 0, -1):
            z = rng.multivariate_normal(np.zeros(28*28), np.eye(28*28))
            z = torch.from_numpy(z)
            z = z.unsqueeze(0)
            alpha_t = (1-betas[t])**t
            y_current, _ = model(x_current)
            x_new = (x_current - y_current/((1 - alpha_t)**0.5))/(alpha_t**0.5) + betas[t]*z
            x_current = x_new
            to_plot = x_current.reshape([28, 28])
            container = ax.imshow(to_plot, cmap='gray')
            artists.append([container])
        ani = animation.ArtistAnimation(fig = fig, artists=artists, interval = 200)
        plt.show()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = 100

best_valid_loss = float('inf')

short = list(islice(train_iterator, 0, 100))

current_train_loss = 1000
previous_train_loss = 1001
epoch = 0

while current_train_loss <= previous_train_loss and epoch < EPOCHS:
    print(f'Starting epoch {epoch}', flush=True)

    start_time = time.monotonic()

    current_train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {current_train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    torch.save(model.state_dict(), f'myfirstdiffusionmodel_{epoch}.pt')
    previous_train_loss = current_train_loss
    epoch += 1



# sample_and_plot(model)

if __name__ == "__main__":
    # pass
    # load the model
    model =  MLP(INPUT_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load("myfirstdiffusionmodel_99.pt", map_location=torch.device('cpu')))
    model.eval()
    betas = np.linspace(1e-4, 1e-1, 10, dtype=np.float64)
    sample_and_plot(model, betas, T=10)

    # Absolutely not working yet. The noise converges to something but it is not an image from the training data.

                          