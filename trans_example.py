from mpi4py import MPI
import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = nn.Linear(input_size, output_size)
grads = {name: param.grad for name, param in model.named_parameters()}
grads_size = sys.getsizeof(pickle.dumps(grads))
state_dict_size = sys.getsizeof(pickle.dumps(model.state_dict()))

if rank == 0:
    model = nn.Linear(input_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grads = {name: param.grad for name, param in model.named_parameters()}

        comm.Send(pickle.dumps(grads), dest=1, tag=200)

        data = bytearray(10000)
        comm.Recv(data, source=1, tag=300)
        remote_state_dict = OrderedDict(pickle.loads(data))
        model.load_state_dict(remote_state_dict)

else:
    model = nn.Linear(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        data = bytearray(10000)
        comm.Recv(data, source=0, tag=200)
        remote_grads = dict(pickle.loads(data))

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            param.grad = remote_grads[name]
        optimizer.step()

        comm.Send(pickle.dumps(model.state_dict()), dest=0, tag=300)
