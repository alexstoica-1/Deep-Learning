# IMPORTS
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import time

# Create the fully-connected NN
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() 
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
"""
TEST:
    model = NN(784, 10)
    x = torch.randn(64, 784)
    print(model(x).shape) # torch.Size([64, 10])
"""

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution with this values 
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # half the dimension size
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
num_epochs_cnn = 5

# Load Data
train_dataset = datasets.MNIST(root = 'neural_network/dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = 'neural_network/dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Networks
model = NN(input_size=input_size, num_classes=num_classes).to(device)
model_cnn = CNN().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =learning_rate)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr =learning_rate)

# Train FeedForward Network

for epoch in range(num_epochs):
    epoch_start = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        data = data.reshape(data.shape[0], -1) # flatens all the rest into a single dimension to get the correct shape 
        #print(data.shape)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets) 

        # Backward
        optimizer.zero_grad() # set all the gradients to 0 for each batch 
        loss.backward()

        # Gradient Descent or Adam step
        optimizer.step()

    epoch_end = time.time()
    print(f"Epoch {epoch+1} took {epoch_end - epoch_start:.2f} seconds.")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data:")
    else:
        print("Checking accuracy on test data:")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}.")

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

# Train Convolutional Network

for epoch in range(num_epochs_cnn):
    epoch_start = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        # Forward
        scores = model_cnn(data)
        loss = criterion(scores, targets) 

        # Backward
        optimizer_cnn.zero_grad() # set all the gradients to 0 for each batch 
        loss.backward()

        # Gradient Descent or Adam step
        optimizer_cnn.step()

    epoch_end = time.time()
    print(f"Epoch {epoch+1} for CNN took {epoch_end - epoch_start:.2f} seconds.")

def check_accuracy_cnn(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data:")
    else:
        print("Checking accuracy on test data:")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}.")

    model.train()

check_accuracy_cnn(train_loader, model_cnn)
check_accuracy_cnn(test_loader, model_cnn)