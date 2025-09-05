import time 
import numpy as np 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# MODEL
class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()

        # 1st hidden layer
        self.linear_1 = nn.Linear(num_features, num_hidden_1)
        # weight initialization
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        # 2nd hidden layer
        self.linear_2 = nn.Linear(num_hidden_1, num_hidden_2)
        # weight initialization
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

        # output layer
        self.linear_out = nn.Linear(num_hidden_2, num_classes)
        # weight initialization
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)

        logits = self.linear_out(out)
        probabilities = F.log_softmax(logits, dim = 1)

        return logits, probabilities 

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
random_seed = 42
learning_rate = 0.1
num_epochs  = 10
batch_size = 64 # industry standard I guess

# Architecture
num_features = 784
num_hidden_1 = 128
num_hidden_2  = 256
num_classes = 10

# Load datasets
train_dataset = datasets.MNIST(root = 'dl_models/dataset/',
                               train = True,
                               transform = transforms.ToTensor(),
                               download = True )

test_dataset = datasets.MNIST(root = 'dl_models/dataset/',
                              train = False,
                              transform = transforms.ToTensor(),
                              download = True)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = batch_size,
                          shuffle = True) # we want the model to see the data in different order each epoch

test_loader = DataLoader(dataset = test_dataset,
                          batch_size = batch_size,
                          shuffle = False) # we want deterministic results during evaluation, good for reproductible evaluation

# Check the size 
for images, labels in train_loader:
    print(f'Image batch dimensions:{images.shape}')
    print(f'Image batch dimensions:{labels.shape}')
    break