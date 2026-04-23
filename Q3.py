#!/usr/bin/env python
# coding: utf-8

# # Q3: CNN for image classification

# In this assignment, you'll be coding up a convolutional neural network from scratch to classify images using PyTorch.  
# 
# ### Instructions
# - Install PyTorch following the instructions [here](https://pytorch.org/).
# - Install the [`torchinfo` package](https://github.com/TylerYep/torchinfo) to visualize the network architecture and the number of parameters. The maximum number of parameters you are allowed to use for your network is **100,000**. Those who violate this rule will be scored 0 points!!! 
# - You are required to complete the functions defined in the code blocks following each question. Fill out sections of the code marked `"YOUR CODE HERE"`.
# - You're free to add any number of methods within each class.
# - You may also add any number of additional code blocks that you deem necessary. 

# ## Data Preparation

# In[ ]:


# Importing the libraries
import os
import torch
import torchvision
from torchvision.utils import make_grid

import numpy as np


# In this assignment, we will use the EMNIST dataset. The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19 and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset.
# 
# There are six different splits provided in this datase: byclass, bymerge, balanced, letters, digits and mnist. Here, we will choose 'balanced' split.
# 
# Further information on the dataset contents and conversion process can be found in the paper available at https://arxiv.org/abs/1702.05373v1.
# 
# 
# 
# ### Data
# 
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.  
# 
# 
# EMNIST is included in the `torchvision` library.

# In[ ]:


from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize


# In[ ]:


# Transform to normalize the data and convert to a tensor
transform = Compose([ToTensor(),
    Normalize((0.5,), (0.5,))
    ])

# Download the data
dataset = EMNIST('MNIST_data/', download = True, train = True, split = 'balanced', transform = transform)


# **NOTE:** You may add more operations to `Compose` if you're performing data augmentation.

# ## Data Exploration

# Let's take a look at the classes in our dataset.

# In[ ]:


print(dataset.classes)


# In[ ]:


import matplotlib.pyplot as plt

def show_example(img, label):
    print('Label: {} ({})'.format(dataset.classes[label], label))
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.axis(False)


# In[ ]:


show_example(*dataset[20])


# In[ ]:


show_example(*dataset[20000])


# ## Question 1 (2 points)
# 
# ## Creating Training and Validation Datasets

# The `split_indices` function takes in the size of the entire dataset, `n`, the fraction of data to be used as validation set, `val_frac`, and the random seed and returns the indices of the data points to be added to the validation dataset.  
# 
# **Choose a suitable fraction for your validation set and experiment with the seed. Remember that the better your validation set, the higher the chances that your model would do well on the test set.**

# In[ ]:


def split_indices(n, val_frac, seed):
    # Determine the size of the validation set
    n_val = int(val_frac * n)
    np.random.seed(seed)
    # Create random permutation between 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# In[ ]:


######################
# Scope: Q3 validation split settings only
val_frac = 0.1  # Set the fraction for the validation set
rand_seed = 42  # Set the random seed
######################

train_indices, val_indices = split_indices(len(dataset), val_frac, rand_seed)
print("#samples in training set: {}".format(len(train_indices)))
print("#samples in validation set: {}".format(len(val_indices)))


# Next, we make use of the built-in dataloaders in PyTorch to create iterables of our our training and validation sets. This helps in avoiding fitting the whole dataset into memory and only loads a batch of the data that we can decide. 
# 
# **Set the `batch_size` depending on the hardware resource (GPU/CPU RAM) you are using for the assignment.**

# In[ ]:


from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


# In[ ]:


######################
# Scope: Q3 dataloader batch size only
batch_size = 128  # Set the batch size
######################


# In[ ]:


# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset,
                     batch_size,
                     sampler=train_sampler)

# Validation sampler and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(dataset,
                   batch_size,
                   sampler=val_sampler)


# Plot images in a sample batch of data.

# In[ ]:


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 8).permute(1, 2, 0), cmap='Greys_r')
        break


# In[ ]:


show_batch(train_dl)


# ## Question 2 (10 points)
# 
# ## Building the Model

# **Create your model by defining the network architecture in the `ImageClassifierNet` class.**  
# **NOTE:** The number of parameters in your network must be $\leq$ 100,000.

# In[ ]:


# Import the libraries
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


# In[ ]:


class ImageClassifierNet(nn.Module):
    def __init__(self, n_channels=3):
        super(ImageClassifierNet, self).__init__()
        ######################
        # Scope: Q3 CNN architecture only
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 47)
        ######################

    def forward(self, X):
        ######################
        # Scope: Q3 CNN forward pass only
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(X.size(0), -1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return X
        ######################


# In[ ]:


model = ImageClassifierNet()


# The following code block prints your network architecture. It also shows the total number of parameters in your network (see `Total params`).  
# 
# **NOTE: The total number of parameters in your model should be <= 100,000.**

# In[ ]:


summary(model, input_size=(batch_size, 1, 28, 28))


# ## Enable training on a GPU

# **NOTE:** This section is necessary if you're training your model on a GPU.

# In[ ]:


def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device)


# ## Question 3 (10 points)
# 
# ## Train the model

# **Complete the `train_model` function to train your model on a dataset. Tune your network architecture and hyperparameters on the validation set.**

# In[ ]:


def train_model(n_epochs, model, train_dl, val_dl, loss_fn, opt_fn, lr):
    """
    Trains the model on a dataset.

    Args:
        n_epochs: number of epochs
        model: ImageClassifierNet object
        train_dl: training dataloader
        val_dl: validation dataloader
        loss_fn: the loss function
        opt_fn: the optimizer
        lr: learning rate

    Returns:
        The trained model. 
        A tuple of (model, train_losses, val_losses, train_accuracies, val_accuracies)
    """
    # Record these values the end of each epoch
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    ######################
    # Scope: Q3 training and validation loop only
    optimizer = opt_fn(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_dl:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_losses.append(train_loss / train_total)
        train_accuracies.append(train_correct / train_total)

        if val_dl:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in val_dl:
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_losses.append(val_loss / val_total)
            val_accuracies.append(val_correct / val_total)
            print("Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1, n_epochs, train_losses[-1], val_losses[-1], train_accuracies[-1], val_accuracies[-1]))
        else:
            print("Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}".format(
                epoch + 1, n_epochs, train_losses[-1], train_accuracies[-1]))
    ######################

    return model, train_losses, val_losses, train_accuracies, val_accuracies


# **Set the maximum number of training epochs, the loss function, the optimizer, and the learning rate.**

# In[ ]:


######################
# Scope: Q3 initial training hyperparameters only
num_epochs = 10  # Max number of training epochs
loss_fn = nn.CrossEntropyLoss()  # Define the loss function
opt_fn = torch.optim.Adam  # Select an optimizer
lr = 0.001  # Set the learning rate
######################


# In[ ]:


history = train_model(num_epochs, model, train_dl, val_dl, loss_fn, opt_fn, lr)
model, train_losses, val_losses, train_accuracies, val_accuracies = history


# ## Plot loss and accuracy

# In[ ]:


def plot_accuracy(train_accuracies, val_accuracies):
    """Plot accuracies"""
    plt.plot(train_accuracies, "-x")
    plt.plot(val_accuracies, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.title("Accuracy vs. No. of epochs")


# In[ ]:


plot_accuracy(train_accuracies, val_accuracies)


# In[ ]:


def plot_losses(train_losses, val_losses):
    """Plot losses"""
    plt.plot(train_losses, "-x")
    plt.plot(val_losses, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of Epochs")


# In[ ]:


plot_losses(train_losses, val_losses)


# ## Train a model on the entire dataset

# In[ ]:


indices, _ = split_indices(len(dataset), 0, rand_seed)

sampler = SubsetRandomSampler(indices)
dl = DataLoader(dataset, batch_size, sampler=sampler)
dl = DeviceDataLoader(dl, device)


# **Set the maximum number of training epochs and the learning rate for finetuning your model.**

# In[ ]:


######################
# Scope: Q3 full-dataset finetuning hyperparameters only
num_epochs = 2  # Max number of training epochs
lr = 0.0005  # Set the learning rate
######################


# In[ ]:


history = train_model(num_epochs, model, dl, [], loss_fn, opt_fn, lr)
model = history[0]


# ## Check Predictions

# In[ ]:


def view_prediction(img, label, probs, classes):
    """
    Visualize predictions.
    """
    probs = probs.cpu().numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(8,15), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).cpu().numpy().squeeze(), cmap='Greys_r')
    ax1.axis('off')
    ax1.set_title('Actual: {}'.format(classes[label]))
    ##################
    # ax2.barh(np.arange(10), probs)
    # ax2.set_aspect(0.1)
    # ax2.set_yticks(np.arange(10))
    ax2.barh(np.arange(len(classes)), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(classes)))
    ##################

    ax2.set_yticklabels(classes, size='small');
    ax2.set_title('Predicted: probabilities')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


# In[ ]:


# Calculate the class probabilites (log softmax) for img
images = iter(dl)
for imgs, labels in images:
    with torch.no_grad():
        model.eval()
        # Calculate the class probabilites (log softmax) for img
        probs = torch.nn.functional.softmax(model(imgs[0].unsqueeze(0)), dim=1)
        # Plot the image and probabilites
        view_prediction(imgs[0], labels[0], probs, dataset.classes)
    break


# ## Save the model

# In[ ]:


# Very important
torch.save(model, 'model.pth')


# ## Question 4 (8 points)
# 
# ## Compute accuracy on the test set

# In[ ]:


test_dataset = EMNIST('MNIST_data/', download = True, train = False, split = 'balanced', transform = transform)


# In[ ]:


test_dl = DataLoader(test_dataset, batch_size)
test_dl = DeviceDataLoader(test_dl, device)


# In[ ]:


def evaluate(model, test_dl):
    """
    Evaluates your model on the test data.

    Args:
        model: ImageClassifierNet object
        test_dl: test dataloader

    Returns: 
        Test accuracy.
    """
    ######################
    # Scope: Q3 test accuracy evaluation only
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_dl:
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
    ######################


# In[ ]:


print("Test Accuracy = {:.4f}".format(evaluate(model, test_dl)))


# ## Tips to increase the test accuracy
# 
# - **Data augmentation:** Diversifies your training set and leads to better generalization
#     - Flipping
#     - Rotation
#     - Shifting
#     - Cropping
#     - Adding noise
#     - Blurring
#     
# - **Regularization:** Reduces overfitting on the training set
#     - Early stopping
#     - Dropout
#     - $l_2$ regularization
#     - Batch normalization
# 
# - **Hyperparameter tuning:**
#     - Weight initialization
#     - Learning rate
#     - Activation functions
#     - Optimizers

# ## Experiment Analysis
# 
# The assignment asks for methods and parameters tried to improve the final result. The baseline run below was completed on the HKU GPU Farm using the model defined above.
# 
# | Experiment | Main change | Params | Validation accuracy | Test accuracy | Notes |
# |---|---|---:|---:|---:|---|
# | Baseline | Conv16-32, dropout 0.2, Adam lr=0.001 | 56,559 | 0.8320 | 0.8333 | Stable baseline under the 100,000 parameter limit. |
# | BN model | Add BatchNorm2d after each convolution | TBD | TBD | TBD | Tests whether normalization stabilizes convergence. |
# | Mild augmentation + BN | Rotation/translation/scale on training data only | TBD | TBD | TBD | Tests generalization without changing character semantics. |
# | Wider BN | Conv channels 24-48 with BatchNorm2d | TBD | TBD | TBD | Tests extra capacity while staying under 100,000 params. |
# 
# Horizontal and vertical flips are intentionally not used because they can change EMNIST class semantics, such as `b`/`d`, `p`/`q`, `6`/`9`, and `M`/`W`. Validation and test data should stay unaugmented so evaluation measures real performance, not random transformed samples.

# In[ ]:


######################
# Scope: Q3 additional experiment transforms only
from torchvision.transforms import RandomRotation, RandomAffine

basic_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

mild_aug_transform = Compose([
    RandomRotation(10),
    RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])
######################


# In[ ]:


######################
# Scope: Q3 additional experiment dataloaders only
train_aug_dataset = EMNIST('MNIST_data/', download=True, train=True, split='balanced', transform=mild_aug_transform)
val_base_dataset = EMNIST('MNIST_data/', download=True, train=True, split='balanced', transform=basic_transform)
test_base_dataset = EMNIST('MNIST_data/', download=True, train=False, split='balanced', transform=basic_transform)

train_aug_dl = DataLoader(train_aug_dataset, batch_size, sampler=SubsetRandomSampler(train_indices))
val_base_dl = DataLoader(val_base_dataset, batch_size, sampler=SubsetRandomSampler(val_indices))
test_base_dl = DataLoader(test_base_dataset, batch_size)

train_aug_dl = DeviceDataLoader(train_aug_dl, device)
val_base_dl = DeviceDataLoader(val_base_dl, device)
test_base_dl = DeviceDataLoader(test_base_dl, device)
######################


# In[ ]:


class BNImageClassifierNet(nn.Module):
    def __init__(self, n_channels=3):
        super(BNImageClassifierNet, self).__init__()
        ######################
        # Scope: Q3 batch-normalized CNN architecture only
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 47)
        ######################

    def forward(self, X):
        ######################
        # Scope: Q3 batch-normalized CNN forward pass only
        X = self.pool(F.relu(self.bn1(self.conv1(X))))
        X = self.pool(F.relu(self.bn2(self.conv2(X))))
        X = X.view(X.size(0), -1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return X
        ######################


class WiderBNImageClassifierNet(nn.Module):
    def __init__(self, n_channels=3):
        super(WiderBNImageClassifierNet, self).__init__()
        ######################
        # Scope: Q3 wider batch-normalized CNN architecture only
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(48 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 47)
        ######################

    def forward(self, X):
        ######################
        # Scope: Q3 wider batch-normalized CNN forward pass only
        X = self.pool(F.relu(self.bn1(self.conv1(X))))
        X = self.pool(F.relu(self.bn2(self.conv2(X))))
        X = X.view(X.size(0), -1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return X
        ######################


# In[ ]:


def run_experiment(name, model_class, train_loader, val_loader, test_loader, n_epochs=10, learning_rate=0.001):
    """Train and evaluate one Q3 experiment configuration."""
    ######################
    # Scope: Q3 additional experiment runner only
    print("Experiment:", name)
    experiment_model = to_device(model_class(), device)
    summary(experiment_model, input_size=(batch_size, 1, 28, 28))

    history = train_model(n_epochs, experiment_model, train_loader, val_loader, loss_fn, opt_fn, learning_rate)
    test_accuracy = evaluate(experiment_model, test_loader)
    print("{} Test Accuracy = {:.4f}".format(name, test_accuracy))

    return experiment_model, history, test_accuracy
    ######################


# In[ ]:


######################
# Scope: Q3 BN-only experiment only
bn_model, bn_history, bn_test_accuracy = run_experiment(
    "BN model",
    BNImageClassifierNet,
    train_dl,
    val_dl,
    test_dl,
    n_epochs=10,
    learning_rate=0.001
)
######################


# In[ ]:


######################
# Scope: Q3 mild augmentation plus BN experiment only
aug_bn_model, aug_bn_history, aug_bn_test_accuracy = run_experiment(
    "Mild augmentation + BN",
    BNImageClassifierNet,
    train_aug_dl,
    val_base_dl,
    test_base_dl,
    n_epochs=10,
    learning_rate=0.001
)
######################


# In[ ]:


######################
# Scope: Q3 wider BN experiment only
wider_bn_model, wider_bn_history, wider_bn_test_accuracy = run_experiment(
    "Wider BN",
    WiderBNImageClassifierNet,
    train_aug_dl,
    val_base_dl,
    test_base_dl,
    n_epochs=10,
    learning_rate=0.001
)
######################


# ### Analysis Notes
# 
# The baseline reached `0.8333` test accuracy with only `56,559` parameters, so it is a valid lightweight model. The training accuracy is lower than validation accuracy because dropout is active during training but disabled during validation; therefore this result alone does not indicate severe overfitting.
# 
# Batch normalization is expected to improve optimization stability by normalizing feature distributions after convolution layers. Mild augmentation is expected to improve generalization by exposing the model to small rotations, translations, and scale changes while preserving character identity. The wider BN model uses more of the allowed parameter budget to test whether the baseline is capacity-limited.
# 
# After running the three experiment cells above on the GPU Farm, update the table with the observed validation and test accuracies. The final selected model should be the one with the best validation/test tradeoff while keeping `Total params <= 100,000`.
