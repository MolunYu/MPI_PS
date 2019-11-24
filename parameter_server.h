//
// Created by yml on 11/23/19.
//

#ifndef PS_MPI_PARAMETER_SERVER_H
#define PS_MPI_PARAMETER_SERVER_H

void gaussian_init(double *params, int params_size);

void sgd(double *params, double *gradient, int param_size, double lr=0.001);

#endif //PS_MPI_PARAMETER_SERVER_H
