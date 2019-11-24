//
// Created by yml on 11/24/19.
//

#ifndef PS_MPI_WORKER_H
#define PS_MPI_WORKER_H

#define params_size 10


class Model {
public:
    Model() { gaussian_init(); }

    ~Model() {}

    void getGradient(double *data, double *labels, double *gradients, int data_len) {
        forward(data);
        backward(labels, gradients);
    }

    double params[params_size];

private:

    double outputs[params_size];

    void forward(double *data) {
        // Compute the data to output
        return;
    }

    void backward(double *label, double *gradients) {
        return;
    }


    void gaussian_init() {
        for (int i = 0; i < params_size; ++i) {
            // Initial with gaussian distribution
            // params[i] = random
        }
    }
};

#endif //PS_MPI_WORKER_H
