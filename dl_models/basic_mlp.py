import time 
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import numpy as np 
import matplotlib.pyplot as plt

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
    #plt.imshow(images[0].squeeze(), cmap="gray")
    #plt.show()
    break

# Network
torch.manual_seed(random_seed)
model = MLP(num_features = num_features,
            num_classes = num_classes)

# Loss function
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

def compute_accuracy(network, data_loader):
    network.eval()
    correct_pred = 0
    num_examples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.contiguous().view(-1, 28*28).to(device)
            targets = targets.to(device)
            logits, probabilities = network(features)
            _, predicted_labels = torch.max(probabilities, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        
        return float(correct_pred) / float(num_examples) * 100
    
start_time = time.time()
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):
        features  = features.contiguous().view(-1, 28*28).to(device)
        targets = targets.to(device)

        # Forward pass
        logits, probabilities = model(features)
        loss = F.cross_entropy(logits, targets)

        #Backwards pass
        optimizer.zero_grad()
        loss.backward()

        # SGD to update the weights 
        optimizer.step()

        # LOGGING
        if not batch_idx % 100:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, len(train_loader), loss))

    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' 
              % (epoch+1, num_epochs, compute_accuracy(model, train_loader)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))