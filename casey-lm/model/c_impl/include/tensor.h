#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

struct Tensor {
    float *data;
    int *strides;
    int *shape;
    int num_dims;
    int size;
    struct BackpropConfig *backprop_config;
};

struct BackpropConfig {
    struct BackpropParent **parents;
    int num_parents;
    float *gradients;
};

struct BackpropParent {
    struct Tensor *tensor;
    float* (*grad_fn)(float*);
};

struct Tensor *tensor_from_data(float *data, int *shape, int num_dims, int size, bool requires_grad);
struct Tensor *tensor_add(struct Tensor *a, struct Tensor *b, bool requires_grad);
void tensor_backward(struct Tensor *t);
void tensor_free(struct Tensor *t);

#endif