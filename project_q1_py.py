#!pip install torch torchvision matplotlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dir = '/home/tehilaoh/project/train'
test_dir = '/home/tehilaoh/project/test'

output_dir = '/home/tehilaoh/project/output1'
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # Fully Connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjusting the size of the input in accordance with the size of the images
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  #2 Categories: buildings and forest

    def forward(self, x):
        # Go through the convolution and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten to transition to fully connected layers
        x = x.view(-1, 128 * 16 * 16)

        # Transition in fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # The output layer
        x = self.fc3(x)

        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5.1.1 Initial loss
def check_initial_loss(model, criterion, data_loader):
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print(f'Initial loss: {loss.item()}')

check_initial_loss(model, criterion, train_loader)

# 5.1.2 Initial loss + regression
def check_initial_loss_with_regression(model, criterion, optimizer, data_loader):
    model.train()
    inputs, labels = next(iter(data_loader))

    # Showing the initial loss
    outputs = model(inputs)
    initial_loss = criterion(outputs, labels)
    print(f'Initial loss before regression: {initial_loss.item()}')

    # Regression step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'Initial loss after one step of regression: {loss.item()}')

    # Second regression step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'Loss after two steps of regression: {loss.item()}')

check_initial_loss_with_regression(model, criterion, optimizer, train_loader)

# 5.1.3 Overfit over few images
def overfit_few_images(model, criterion, optimizer, num_epochs=100):
    model.train()
    inputs, labels = next(iter(train_loader))

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Loss after overfitting few images: {loss.item()}')

    # Switch to evaluation mode to test the performance on the same images
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        final_loss = criterion(outputs, labels)
        print(f'Final loss on the same images after overfitting: {final_loss.item()}')

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / len(labels) * 100
        print(f'Final Accuracy on same images: {accuracy:.2f}%')

overfit_few_images(model, criterion, optimizer)

def save_plot(fig, filename, title):
    fig.suptitle(title)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def validate_model(model, criterion, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10, experiment_name="experiment"):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update the weights

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validate the model on the test set
        test_loss, test_acc = validate_model(model, criterion, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    # Plotting loss and accuracy side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Train and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Train and Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    # Saving the combined plot
    save_plot(fig, f'{experiment_name}_loss_accuracy.png', f'{experiment_name} - Training and Testing Loss and Accuracy')

def show_images(model, data_loader, num_images=5, experiment_name="experiment"):
    model.eval()

    correct_images_buildings = []
    correct_images_forest = []
    incorrect_images_buildings = []
    incorrect_images_forest = []
    correct_labels_buildings = []
    correct_labels_forest = []
    incorrect_labels_buildings = []
    incorrect_labels_forest = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(inputs)):
                if labels[i].item() == 0:  # Assuming 0 is the label for buildings
                    if predicted[i] == labels[i] and len(correct_images_buildings) < num_images:
                        correct_images_buildings.append(inputs[i])
                        correct_labels_buildings.append(predicted[i])
                    elif predicted[i] != labels[i] and len(incorrect_images_buildings) < num_images:
                        incorrect_images_buildings.append(inputs[i])
                        incorrect_labels_buildings.append(predicted[i])
                elif labels[i].item() == 1:  # Assuming 1 is the label for forest
                    if predicted[i] == labels[i] and len(correct_images_forest) < num_images:
                        correct_images_forest.append(inputs[i])
                        correct_labels_forest.append(predicted[i])
                    elif predicted[i] != labels[i] and len(incorrect_images_forest) < num_images:
                        incorrect_images_forest.append(inputs[i])
                        incorrect_labels_forest.append(predicted[i])

            if (len(correct_images_buildings) >= num_images and len(incorrect_images_buildings) >= num_images and
                len(correct_images_forest) >= num_images and len(incorrect_images_forest) >= num_images):
                break

    # Display and save correct and incorrect images together
    fig, axs = plt.subplots(4, num_images, figsize=(15, 30))

    # Correct building images
    for i in range(len(correct_images_buildings)):
        axs[0, i].imshow(correct_images_buildings[i].permute(1, 2, 0))
        axs[0, i].set_title(f'Correct building: {correct_labels_buildings[i].item()}')
        axs[0, i].axis('off')

    # Incorrect building images
    for i in range(len(incorrect_images_buildings)):
        axs[1, i].imshow(incorrect_images_buildings[i].permute(1, 2, 0))
        axs[1, i].set_title(f'Incorrect building: {incorrect_labels_buildings[i].item()}')
        axs[1, i].axis('off')

    # Correct forest images
    for i in range(len(correct_images_forest)):
        axs[2, i].imshow(correct_images_forest[i].permute(1, 2, 0))
        axs[2, i].set_title(f'Correct forest: {correct_labels_forest[i].item()}')
        axs[2, i].axis('off')

    # Incorrect forest images
    for i in range(len(incorrect_images_forest)):
        axs[3, i].imshow(incorrect_images_forest[i].permute(1, 2, 0))
        axs[3, i].set_title(f'Incorrect forest: {incorrect_labels_forest[i].item()}')
        axs[3, i].axis('off')

    # Save the combined plot
    save_plot(fig, f'{experiment_name}_correct_incorrect_images.png', f'{experiment_name} - Correct and Incorrect Classified Images')

# Experiment with different learning rates
for lr in [0.01, 0.001, 0.0001]:
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    experiment_name = f'learning_rate_{lr}'
    print(f'Training with learning rate: {lr}')
    train_model(model, criterion, optimizer, train_loader, test_loader, experiment_name=experiment_name)
    show_images(model, test_loader, experiment_name=experiment_name)

# Experiment with different batch sizes
for batch_size in [16, 32, 64]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    experiment_name = f'batch_size_{batch_size}'
    print(f'Training with batch size: {batch_size}')
    train_model(model, criterion, optimizer, train_loader, test_loader, experiment_name=experiment_name)
    show_images(model, test_loader, experiment_name=experiment_name)

# Experiment with different optimizations
for opt in [optim.Adam, optim.SGD, optim.RMSprop]:
    model = SimpleCNN()
    optimizer = opt(model.parameters(), lr=0.001)
    experiment_name = f'optimizer_{opt.__name__}'
    print(f'Training with optimizer: {opt.__name__}')
    train_model(model, criterion, optimizer, train_loader, test_loader, experiment_name=experiment_name)
    show_images(model, test_loader, experiment_name=experiment_name)

# Experiment with data augmentation
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset_aug = datasets.ImageFolder(train_dir, transform=transform_augment)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment_name = 'data_augmentation'
print('Training with data augmentation')
train_model(model, criterion, optimizer, train_loader_aug,test_loader, experiment_name=experiment_name)
show_images(model, test_loader, experiment_name=experiment_name)

# Experiment without data augmentation
transform_no_augment = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset_no_aug = datasets.ImageFolder(train_dir, transform=transform_no_augment)
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=32, shuffle=True)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment_name = 'no_data_augmentation'
print('Training without data augmentation')
train_model(model, criterion, optimizer, train_loader_no_aug, test_loader, experiment_name=experiment_name)
show_images(model, test_loader, experiment_name=experiment_name)

# Experiment with different grid sizes
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(0.5)

        #  Fully Connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        # Go through the convolution and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten to transition to fully connected layers
        x = x.view(-1, 256 * 8 * 8)

        # Transition in fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)

        return x

model = EnhancedCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment_name = 'large_network'
print('Training with large network')
train_model(model, criterion, optimizer, train_loader, test_loader, experiment_name=experiment_name)
show_images(model, test_loader, experiment_name=experiment_name)

# Experiment with different image sizes
transform_small = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset_small = datasets.ImageFolder(train_dir, transform=transform_small)
train_loader_small = DataLoader(train_dataset_small, batch_size=32, shuffle=True)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment_name = 'small_images'
print('Training with small images')
train_model(model, criterion, optimizer, train_loader_small,test_loader, experiment_name=experiment_name)
show_images(model, test_loader, experiment_name=experiment_name)

#  Experiment Change number of neurons in fully connected layer
class ModifiedCNN(nn.Module):
    def __init__(self, fc1_neurons=128):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 30 * 30, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

for neurons in [256, 512, 1024]:
    model = ModifiedCNN(fc1_neurons=neurons)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    experiment_name = f'fc1_neurons_{neurons}'
    print(f'Training with {neurons} neurons in fully connected layer')
    train_model(model, criterion, optimizer, train_loader, test_loader, experiment_name=experiment_name)
    show_images(model, test_loader, experiment_name=experiment_name)

# Experiment Changing number of epochs
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment_name = 'epochs_20'
print('Training with 20 epochs')
train_model(model, criterion, optimizer, train_loader,test_loader, epochs=20, experiment_name=experiment_name)
show_images(model, test_loader, experiment_name=experiment_name)