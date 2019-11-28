from mpi4py import MPI
import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict

# Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
server_rank = worker_size = size - 1

# tag
tag_gradient_trans = 0
tag_params_trans = 1

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
trans_size = sys.getsizeof(pickle.dumps(model.state_dict()))

if rank == server_rank:
    model = nn.Linear(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        data = [bytearray(trans_size)] * worker_size
        recv_request = [None] * worker_size
        send_request = [None] * worker_size

        for i in range(worker_size):
            recv_request[i] = comm.Irecv(data[i], source=0, tag=tag_gradient_trans)

        MPI.Request.Waitall(recv_request)

        grads = [dict(pickle.loads(i)) for i in data]

        optimizer.zero_grad()
        for name, param in model.named_parameters():
            param.grad = sum([grad[name] for grad in grads])
        optimizer.step()

        for i in range(worker_size):
            send_request[i] = comm.Isend(pickle.dumps(model.state_dict()), dest=0, tag=tag_params_trans)
        MPI.Request.Waitall(send_request)

else:
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
        grads = {name: param.grad for name, param in model.named_parameters()}

        comm.Send(pickle.dumps(grads), dest=1, tag=tag_gradient_trans)

        data = bytearray(trans_size)
        comm.Recv(data, source=1, tag=tag_params_trans)
        remote_state_dict = OrderedDict(pickle.loads(data))
        model.load_state_dict(remote_state_dict)
