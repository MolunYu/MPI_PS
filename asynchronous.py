import torch
import sys
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from mpi4py import MPI
from collections import OrderedDict
import threading
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
server_rank = worker_size = size - 1
lock = threading.Lock()
start = time.time()

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


class SynchronizeThread(threading.Thread):
    def __init__(self, model, optimizer, worker_rank, epochs, steps):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.worker_rank = worker_rank
        self.epochs = epochs
        self.steps = steps

    def run(self) -> None:
        recv_buf = bytearray(trans_size)

        for epoch in range(self.epochs):
            for step in range(self.steps):
                send_request = comm.isend(MPI.pickle.dumps(self.model.state_dict()), dest=self.worker_rank,
                                          tag=tag_params_trans)

                MPI.Request.Wait(send_request)

                recv_request = comm.irecv(recv_buf, source=self.worker_rank, tag=tag_gradient_trans)
                data = MPI.Request.wait(recv_request)

                grad = dict(MPI.pickle.loads(data))

                lock.acquire()
                self.optimizer.zero_grad()
                for name, param in model.named_parameters():
                    param.grad = grad[name]
                self.optimizer.step()
                lock.release()


model = NeuralNet(input_size, hidden_size, num_classes)
trans_size = sys.getsizeof(MPI.pickle.dumps(model.state_dict()))
del model

if rank == server_rank:
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    synchronize_list = list()
    for i in range(worker_size):
        synchronize_list.append(SynchronizeThread(model, optimizer, i, num_epochs, len(train_loader)))

    for t in synchronize_list:
        t.start()

    for t in synchronize_list:
        t.join()

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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            data = comm.recv(source=server_rank, tag=tag_params_trans)
            remote_state_dict = OrderedDict(MPI.pickle.loads(data))
            model.load_state_dict(remote_state_dict)

            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            model.zero_grad()
            loss.backward()

            grads = {name: param.grad for name, param in model.named_parameters()}
            comm.send(MPI.pickle.dumps(grads), dest=server_rank, tag=tag_gradient_trans)

            if (i + 1) % 100 == 0:
                print('{} rank {} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(time.time() - start, rank, epoch + 1, num_epochs, i + 1, total_step, loss.item()))
