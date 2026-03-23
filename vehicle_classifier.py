import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Written By: Wafa Jailani
# Email: wafajailani407@gmail.com
# GitHub Repository:
# GitHub Account: @wafajailani

#there are commented print statements spread throughout to verify the dataset dimensions and parameters, ratio of 8:2, and image predictions compared to GroundTruth
#loss and final accuracy are printed

# images are 3-channel (RBD) and 64x64
# 7 different vehicle types, 1 nonvehicle
# count for each label is ~750-9000

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # keep at 64 by 64

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 1. Load the data

dataset = ImageFolder(root="C:/UTDLabCodingChallenge/vehicle_classification", transform=transform)

"""
print("\n--- Load Data ---")
print("Classes: ", dataset.classes)
print("Class → Index mapping:", dataset.class_to_idx)
print("Total images found:", len(dataset))
"""

# 2. Divide the dataset into a training set and a testing set with a ratio of 8:2.

# training set is the primary data used to learn patterns
# validation set is more of fine-tuning and optimization
# testing set measures final performance

total_size = len(dataset)
train_size = int(0.8 * total_size)  # 8
test_size = total_size - train_size  # 2

torch.manual_seed(42)
training_data, test_data = random_split(dataset, [train_size, test_size])

# DataLoader wraps an iterable on Dataset
train_dataloader = DataLoader(
    training_data,
    batch_size=128,  # 64
    shuffle=True,
    num_workers=0
)

test_dataloader = DataLoader(
    test_data,
    batch_size=128,  # 64
    shuffle=True,
    num_workers=0
)
images, labels = next(iter(train_dataloader))


"""
print("\n--- Verification ---")
print("Batch shape:  ", images.shape)  # torch.Size([64, 3, 64, 64])
print("Labels shape: ", labels.shape)  # torch.Size([64])
print("Min pixel:    ", images.min().item())  # around -1.0
print("Max pixel:    ", images.max().item())  # around +1.0
print("\nData loaded successfully — ready to train!")

print("\n --- Check 8:2 Ratio ---")
print("Training set size:  ", len(training_data))
print("Testing set size:   ", len(test_data))

print(f"Train ratio: {len(training_data) / len(dataset) * 100:.2f}%")
print(f"Test ratio:  {len(test_data) / len(dataset) * 100:.2f}%")
print()
"""

# Steps 3 and 4 below

# 3. Build a CNN model that takes an image (or a batch of images) as input and output the class that the input image(s) are belonging to
# 4. Train the model and keep track of the loss and accuracy.
import torch.optim as optim

num_epochs = 10  # high loss with 4 epochs, the more epochs the better/lower loss value
learning_rate = 0.001

""" 
Training Session #1 
    epochs = 4
    lr = 0.001
    batch_size = 64
    reps = 100
    -> loss = 1.894

Training Session #2 (2 runs)
    epochs = 4
    lr = 0.0001
    batch_size = 128
    reps = 100
    -> loss = 2.018, 2.184

Training Session #3 
    epochs = 10
    lr = 0.001
    batch_size = 128
    reps - 50
    -> loss = 0.508 (switched optimizer to Adam)

Training Session #4 (2 runs)
    epochs = 10
    lr = 0.001
    batch_size = 128
    reps = 50
    -> loss = 0.508, 0.468
    -> accuracy = 75%, 75%

Training Session #5 (4 runs)
    epochs = 10
    lr = 0.001
    batch_size = 128
    reps = 50
    -> loss = 0.508, 0.128, 0.046, 0.020
    

Training Session #6 (4 runs)
    epochs = 10
    lr = 0.001
    batch_size = 128
    reps = 50
    -> loss = 1.491, 0.418, 0.152, 0.084
    -> accuracy = 75%, 77%, 74%, 75% 
    

Training Session #7 
    epochs = 10
    lr = 0.001
    batch_size = 128
    reps = 50
    -> loss = 0.018
    -> accuracy = 75%
    
    In training session 7, by epoch 10 the loss was 0.018 with an accuracy of 75%

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # 16*13*13 =
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net().to(device)


n_total_steps = len(train_dataloader)

import os

PATH = './vehicle_classifier.pth'

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

if os.path.exists(PATH):
    checkpoint = torch.load(PATH, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded saved model — continuing training")
else:
    print("No saved model found — training from scratch")

print("\nPrinting Epochs, Batch, and Loss...")

for epoch in range(num_epochs):  # loop over the dataset num_epochs amount of times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print stats and loss
        running_loss += loss.item()
        if i % 50 == 49:  # print every 50 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            # running_loss = 0.0

print("EXIT")

# save model AFTER training
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, PATH)
#print(f"Model saved to {PATH}")

dataiter = iter(test_dataloader)
images, labels = next(dataiter)

classes = dataset.classes


def imshow(img):
    img = img / 2 + 0.5  # unnormalize — reverses the normalization we did earlier
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#comparing GroundTruth and what CNN predict to see if model working right
"""imshow(torchvision.utils.make_grid(images[:4].cpu()))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = net(images.to(device))

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
                              
"""

# 5. Print the final accuracy.

correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)  # uses class with the highest output energy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# inside training loop, after all batches finish
epoch_accuracy = 100 * correct / total
#print(f'Epoch [{epoch+1}/{num_epochs}] Accuracy: {epoch_accuracy:.2f}%') #probably should be in the loop

print(f'Accuracy of the network on the {total} test images: {100 * correct // total}%')

# 6. (Optional bonus task) Plot the loss and accuracy curve during the training.
# attempted here but ended up plotting manually

"""

import json

STATS_PATH = './training_stats.json'

if os.path.exists(STATS_PATH):
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    train_losses     = stats['losses']
    train_accuracies = stats['accuracies']
    print(f"Loaded stats — {len(train_losses)} epochs recorded so far")
else:
    train_losses     = []
    train_accuracies = []
    print("Starting fresh stats")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(1, len(train_losses)+1), train_losses, color='red', marker='o')
ax1.set_title('Training Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(range(1, len(train_accuracies)+1), train_accuracies, color='blue', marker='o')
ax2.set_title('Training Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.suptitle(f'Loss vs Accuracy — {len(train_losses)} total epochs', fontsize=14)
plt.tight_layout()
plt.show()

"""