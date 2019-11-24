#include <iostream>
#include "mpi.h"
#include "parameter_server.h"
#include "worker.h"

#define tag_exit 0
#define tag_gradient_trans 1
#define tag_params_trans 2

int main(int argc, char *argv[]) {
    int rank, size, worker_size, server_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    server_rank = worker_size = size - 1;

    if (rank == server_rank) {
        double server_params[params_size];
        gaussian_init(server_params, params_size);

        int exit_flag = 0;
        int *exit_code = new int[worker_size];
        MPI_Request *exit_request = new MPI_Request[worker_size];
        MPI_Status *exit_status = new MPI_Status[worker_size];

        int update_rank;
        double **gradients = new double *[worker_size];

        for (int i = 0; i < worker_size; ++i) {
            gradients[i] = new double[params_size];
        }

        MPI_Request *gradient_recv_request = new MPI_Request[worker_size];
        MPI_Status *gradient_recv_status = new MPI_Status[worker_size];

        for (int i = 0; i < worker_size; ++i) {
            MPI_Irecv(&exit_code[i], 1, MPI_INT, i, tag_exit, MPI_COMM_WORLD, &exit_request[i]);

            MPI_Recv_init(gradients[i], params_size, MPI_DOUBLE, i, tag_gradient_trans, MPI_COMM_WORLD,
                          &gradient_recv_request[i]);
        }

        MPI_Startall(worker_size, gradient_recv_request);

        while (MPI_Testall(worker_size, exit_request, &exit_flag, exit_status) != MPI_SUCCESS) {
            MPI_Waitany(worker_size, gradient_recv_request, &update_rank, gradient_recv_status);
            sgd(server_params, gradients[update_rank], params_size);
            MPI_Send(server_params, params_size, MPI_DOUBLE, update_rank, tag_params_trans, MPI_COMM_WORLD);

            MPI_Start(&gradient_recv_request[update_rank]);
        }

        for (int i = 0; i < worker_size; ++i) {
            MPI_Request_free(&gradient_recv_request[i]);
        }

        delete[] exit_code;
        delete[] exit_request;
        delete[] exit_status;
        delete[] gradients;
        delete[] gradient_recv_request;
        delete[] gradient_recv_status;
    } else {
        Model model;
        MPI_Status status;
        double gradients[params_size];

        int epoch = 100;
        int data_size = 32;
        int data_len = 30;
        double **data = new double*[data_size];
        double **label = new double*[data_size];
        for (int i = 0; i < data_size; ++i) {
            data[i] = new double[data_len];
            label[i] = new double[data_len];
        }

        for (int i = 0; i < epoch; ++i) {
            for (int j = 0; j < data_size; ++j) {
                model.getGradient(data[i], label[i], gradients, data_len);
                MPI_Send(gradients, params_size, MPI_DOUBLE, server_rank, tag_gradient_trans, MPI_COMM_WORLD);
                MPI_Recv(model.params, params_size, MPI_DOUBLE, server_rank, tag_params_trans, MPI_COMM_WORLD, &status);
            }
        }

    }


    MPI_Finalize();

    return 0;
}