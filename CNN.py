import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading preprocessed data
inputs = np.load("inputs.npy") # shape: (4991, 2603, 20)
inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]))  # (4991, 1, 2603, 20)
labels = np.load("labels.npy") # shape: (4991,)


#train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2)
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
# y_train_tensor = F.one_hot(y_train_tensor, num_classes=10)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)
# y_test_tensor = F.one_hot(y_test_tensor, num_classes=10)


class CNN_genre(nn.Module):
    def __init__(self):
        super(CNN_genre, self).__init__()
        # first convolution layer - output shape: (32, 2601, 18)
        self.conv1 = nn.Conv2d(1, 32, (3, 3)) # input channel, output channel (number of kernels), kernel size
        # first two max pooling layer - output shape: (32, 1300, 9)
        self.pool = nn.MaxPool2d((2, 2), (2, 2)) # kernel size, stride
        # second convolution layer - output shape: (32, 1299, 8); after max pooling (32, 649, 4)
        self.conv2 = nn.Conv2d(32, 32, (2, 2))

        # # third convolution layer - output shape: (32, 648, 3); after max pooling (32, 324, 1)
        # self.conv3 = nn.Conv2d(32, 32, (2, 2))

        # first fully connected layer
        self.flat1 = nn.Linear(32*649*4, 512)
        # second fully connected layer
        self.flat2 = nn.Linear(512, 128)
        # third fully connected layer
        self.flat3 = nn.Linear(128, 10) # output size 10 for 10 genres

    def forward(self, x):
        # input shape: (batch size, 1, 2603, 20)
        # first convolution layer with RELU as activation function
        x = F.relu(self.conv1(x))
        # max pooling layer
        x = self.pool(x)
        # second convolution layer with RELU
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # # third convolution layer with RELU
        # x = F.relu(self.conv3(x))
        # x = self.pool(x)

        # flatten before passing into the linear layer
        x = x.view(-1, 32*649*4)
        # first fully connected linear layer with RELU
        x = F.relu(self.flat1(x))
        # second fully connected linear layer with RELU
        x = F.relu(self.flat2(x))
        # final linear layer that will be sent to softmax
        x = self.flat3(x)
        return x

# Training
cnn = CNN_genre().to(device)
batch_size = 128
num_epoch = 30
learning_rate = 0.0002
criterion = nn.CrossEntropyLoss() # use cross entropy loss; softmax is included here
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate) # Stochastic Gradient Descent optimizer


train_data = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(train_data, batch_size=batch_size)
# training
for i in range(num_epoch):
    print("epoch ", i)
    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        # output from the cnn model
        out = cnn(X)
        # compute loss
        loss = criterion(out, y)
        # backpropagation and optimization
        optimizer.zero_grad()  # set gradients to zero
        loss.backward()
        optimizer.step()
print("Training done!")

# save the trained model
torch.save(cnn.state_dict(), 'cnn.pth')


# Testing # evaluation
cnn.eval()  # Set the model to evaluation mode
test_data = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=batch_size)
correct = 0
total = 0

# on training set
print("evaluation on training set:")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for input, labels in dataloader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = cnn(input)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Training accuracy of the network: {acc} %')

    genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hiphop', 'country', 'jazz']
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Training accuracy of {genres[i]}: {acc} %')



# on test set
print("evaluation on test set:")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for input, labels in test_loader:
        input = input.to(device)
        labels = labels.to(device)
        outputs = cnn(input)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Test accuracy of the network: {acc} %')

    genres = ['pop', 'metal', 'disco', 'blues', 'reggae', 'classical', 'rock', 'hiphop', 'country', 'jazz']
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Test accuracy of {genres[i]}: {acc} %')