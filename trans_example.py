import torch
import sys
import torch.nn as nn
import pickle
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from mpi4py import MPI
from collections import OrderedDict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
server_rank = worker_size = size - 1

# tag
tag_gradient_trans = 0
tag_params_trans = 1

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# MNIST dataset
global_dataset = torchvision.datasets.MNIST(root='./',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)
local_dataset_len = len(global_dataset) // worker_size

indices = rank if rank != server_rank else 0

train_dataset = data.Subset(global_dataset, range(local_dataset_len * indices, local_dataset_len * (indices + 1)))
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
trans_size = sys.getsizeof(pickle.dumps(model.state_dict()))

if rank == server_rank:
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data = [bytearray(trans_size)] * worker_size
    recv_request = [None] * worker_size
    send_request = [None] * worker_size

    for epoch in range(num_epochs):
        for _ in train_loader:
            for i in range(worker_size):
                send_request[i] = comm.Isend(pickle.dumps(model.state_dict()), dest=i, tag=tag_params_trans)
            MPI.Request.Waitall(send_request)

            for i in range(worker_size):
                recv_request[i] = comm.Irecv(data[i], source=i, tag=tag_gradient_trans)
            MPI.Request.Waitall(recv_request)

            grads = [dict(pickle.loads(i)) for i in data]

            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = sum([grad[name] for grad in grads])
            optimizer.step()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

else:
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    data = bytearray(trans_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            comm.Recv(data, source=server_rank, tag=tag_params_trans)
            remote_state_dict = OrderedDict(pickle.loads(data))
            model.load_state_dict(remote_state_dict)

            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            grads = {name: param.grad for name, param in model.named_parameters()}

            comm.Send(pickle.dumps(grads), dest=server_rank, tag=tag_gradient_trans)
