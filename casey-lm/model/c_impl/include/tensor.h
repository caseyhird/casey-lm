#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

struct tensor {
    float *data;
    int *strides;
    int *shape;
    int num_dims;
    int size;
    struct backprop_config *backprop_config;
};

struct backprop_config {
    struct backprop_parent **parents;
    int num_parents;
    float gradient;
};

struct backprop_parent {
    struct tensor *tensor;
    float (*grad_fn)(float);
};

struct tensor *tensor_from_data(float *data, int *shape, int num_dims, int size, bool requires_grad);
void tensor_free(struct tensor *t);

#endif