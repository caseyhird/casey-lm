#include "tensor.h"
#include <stdbool.h>
#include <stdlib.h>

struct tensor *tensor_from_data(float *data, int *shape, int num_dims, int size, bool requires_grad) {
    struct tensor *t = malloc(sizeof(struct tensor));
    t->data = data;
    t->shape = shape;
    t->num_dims = num_dims;
    t->size = size;

    t->strides = malloc(num_dims * sizeof(int));
    int stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= shape[i];
    }

    if (requires_grad) {
        t->backprop_config = malloc(sizeof(struct backprop_config));
        t->backprop_config->parents = NULL;
        t->backprop_config->num_parents = 0;
        t->backprop_config->gradient = 0;
    }
    return t;
}

void tensor_free(struct tensor *t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}