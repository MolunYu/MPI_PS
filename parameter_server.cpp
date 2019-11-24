//
// Created by yml on 11/23/19.
//

#include <random>

void gaussian_init(double *params, int params_size) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::normal_distribution<> norm(0, 1);

    for (int i = 0; i < params_size; ++i) {
        params[i] = norm(e);
    }
}
