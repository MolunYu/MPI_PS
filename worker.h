//
// Created by yml on 11/24/19.
//

#ifndef PS_MPI_WORKER_H
#define PS_MPI_WORKER_H

#include <random>

#define params_size 10


class Model {
public:
    Model() { gaussian_init(); }

    ~Model() {}

    void getGradient(double *data, double *labels, double *gradients, int data_len) {
        forward(data, data_len);
        backward(labels, gradients);
    }

    double params[params_size];

private:

    double outputs[params_size];

    virtual void forward(double *data, int data_len) = 0;

    virtual void backward(double *label, double *gradients) = 0;


    void gaussian_init() {
        std::random_device rd;
        std::default_random_engine e(rd());
        std::normal_distribution<> norm(0, 1);

        for (int i = 0; i < params_size; ++i) {
            params[i] = norm(e);
        }
    }
};


// Your model should be create as below:
class MyModel : public Model {
    void forward(double *data, int data_len) override {
        return;
    }

    void backward(double *label, double *gradients) override {
        return;
    }
};


#endif //PS_MPI_WORKER_H
