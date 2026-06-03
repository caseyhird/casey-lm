#include "tensor.h"
#include <stdbool.h>
#include <stdlib.h>


/* Start Private helpers */

static struct Tensor *make_tensor(float *data, int *shape, int num_dims, int size, struct BackpropConfig *backprop_config) {
    struct Tensor *t = malloc(sizeof(struct Tensor));
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

    t->backprop_config = backprop_config;
    return t;
}

static struct BackpropParent *make_backprop_parent(struct Tensor *tensor, float (*grad_fn)(float)) {
    struct BackpropParent *parent = malloc(sizeof(struct BackpropParent));
    parent->tensor = tensor;
    parent->grad_fn = grad_fn;
    return parent;
}

static void backward_inner(struct Tensor *t, float* grad) {
    for (int i = 0; i < t->size; i++) {
        t->backprop_config->gradients[i] += grad[i];
    }
    for (int i = 0; i < t->backprop_config->num_parents; i++) {
        struct BackpropParent *parent = t->backprop_config->parents[i];
        float *parent_grad = parent->grad_fn(t->backprop_config->gradients);
        backward_inner(parent->tensor, parent_grad);
        free(parent_grad);
    }
}

/* End Private helpers */

/* Start Public API */
struct Tensor *tensor_from_data(float *data, int *shape, int num_dims, int size, bool requires_grad) {
    struct BackpropConfig *backprop_config;
    if (requires_grad) {
        backprop_config = malloc(sizeof(struct BackpropConfig));
        backprop_config->parents = NULL;
        backprop_config->num_parents = 0;
        backprop_config->gradients = calloc(size, sizeof(float));
    } else {
        backprop_config = NULL;
    }

    return make_tensor(data, shape, num_dims, size, backprop_config);
}

float *grad_fn_add(float* grad) {
    return grad;
}

struct Tensor *tensor_add(struct Tensor *a, struct Tensor *b, bool requires_grad) {
    // Check that the tensors have the same shape
    if (a->num_dims != b->num_dims || a->size != b->size) {
        return NULL;
    }
    for (int i = 0; i < a->num_dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            return NULL;
        }
    }

    float *data = malloc(a->size * sizeof(float));
    for (int i = 0; i < a->size; i++) {
        data[i] = a->data[i] + b->data[i];
    }

    struct BackpropConfig *backprop_config;
    if (requires_grad) {
        backprop_config = malloc(sizeof(struct BackpropConfig));
        backprop_config->parents = malloc(2 * sizeof(struct BackpropParent));
        backprop_config->num_parents = 2;
        backprop_config->gradients = calloc(a->size, sizeof(float));
        backprop_config->parents[0] = make_backprop_parent(a, grad_fn_add);
        backprop_config->parents[1] = make_backprop_parent(b, grad_fn_add);
    } else {
        backprop_config = NULL;
    }
    return make_tensor(data, a->shape, a->num_dims, a->size, backprop_config);
}

void tensor_backward(struct Tensor *t) {
    if (t->backprop_config == NULL) {
        return;
    }

    backward_inner(t, 1);
}

void tensor_free(struct Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    if (t->backprop_config) {
        for (int i = 0; i < t->backprop_config->num_parents; i++) {
            free(t->backprop_config->parents[i]);
        }
        free(t->backprop_config->parents);
        free(t->backprop_config);
    }
    free(t);
}

/* End Public API */
