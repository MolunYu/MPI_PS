import torch
import sys
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import time
from mpi4py import MPI
from collections import OrderedDict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
server_rank = worker_size = size - 1
synchronize_interval = 0.02
accumulate = 0

# tag
tag_gradient_trans = 0
tag_params_trans = 1

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 128
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


model = NeuralNet(input_size, hidden_size, num_classes)
trans_size = sys.getsizeof(MPI.pickle.dumps(model.state_dict()))
del model

if rank == server_rank:
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recv_request = [None] * worker_size
    send_request = [None] * worker_size
    data = [None] * worker_size
    recv_buf = bytearray(trans_size)
    synchronize_time = time.time()
    count_step = 0
    exit_flag = False

    while True:
        time.sleep(synchronize_interval)
        for i in range(worker_size):
            send_request[i] = comm.isend(MPI.pickle.dumps(model.state_dict()), dest=i, tag=tag_params_trans)
        MPI.Request.Waitall(send_request)

        for i in range(worker_size):
            recv_request[i] = comm.irecv(recv_buf, source=i, tag=tag_gradient_trans)
            data[i] = MPI.Request.wait(recv_request[i])

        grads = [dict(MPI.pickle.loads(i)) for i in data]

        for grad in grads:
            if 'exit' in grad.keys():
                exit_flag = True

        if exit_flag:
            for i in range(worker_size):
                if 'exit' not in grads[i].keys():
                    exit_request = comm.isend(MPI.pickle.dumps({'exit' : 0}), dest=i, tag=tag_params_trans)
                    MPI.Request.Wait(exit_request)
            break

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            param.grad = torch.mean(torch.stack([grad[name] for grad in grads]), dim=0)
        optimizer.step()
        count_step += 1
        print("synchronized {} times".format(count_step))

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
    synchronize_time = time.time()
    exit_flag = False
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            accumulate += 1

            if time.time() - synchronize_time > synchronize_interval:
                data = comm.recv(source=server_rank, tag=tag_params_trans)
                if 'exit' in MPI.pickle.loads(data):
                    exit_flag = True
                    break
                remote_state_dict = OrderedDict(MPI.pickle.loads(data))
                model.load_state_dict(remote_state_dict)

                grads = {name: param.grad / accumulate for name, param in model.named_parameters()}
                comm.send(MPI.pickle.dumps(grads), dest=server_rank, tag=tag_gradient_trans)
                model.zero_grad()
                accumulate = 0
                synchronize_time = time.time()

            # if (i + 1) % 100 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        if exit_flag:
            break

    if not exit_flag:
        data = comm.recv(source=server_rank, tag=tag_params_trans)
        if 'exit' not in MPI.pickle.loads(data):
            comm.send(MPI.pickle.dumps({'exit': 0}), dest=server_rank, tag=tag_gradient_trans)

    print("rank {} Finalize".format(rank))
