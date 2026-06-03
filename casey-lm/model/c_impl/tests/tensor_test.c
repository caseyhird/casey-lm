#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor.h"

/* Start helper functions */

static struct Tensor *make_tensor() {
    float *data = malloc(10 * sizeof(float));
    for (int i=0; i<10; i++) {
        data[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    int *shape = malloc(2 * sizeof(int));
    shape[0] = 2;
    shape[1] = 5;
    int num_dims = 2;
    int size = 10;

    return tensor_from_data(data, shape, num_dims, size, true); 
}

/* End helper functions */

void test_tensor_from_data() {
    float *data = malloc(10 * sizeof(float));
    for (int i=0; i<10; i++) {
        data[i] = i + 1;
    }
    int *shape = malloc(2 * sizeof(int));
    shape[0] = 2;
    shape[1] = 5;
    int num_dims = 2;
    int size = 10;

    struct Tensor *t = tensor_from_data(data, shape, num_dims, size, true);
    
    assert(t->data == data);
    assert(t->shape == shape);
    assert(t->num_dims == num_dims);
    assert(t->size == size);
    assert(t->strides[0] == 5);
    assert(t->strides[1] == 1);
    assert(t->backprop_config->parents == NULL);
    assert(t->backprop_config->num_parents == 0);
    assert(t->backprop_config->gradient == 0);
    tensor_free(t);
}

void test_tensor_add() {
    float *data_a = malloc(10 * sizeof(float));
    for (int i=0; i<10; i++) {
        data_a[i] = i + 1;
    }
    float *data_b = malloc(10 * sizeof(float));
    for (int i=0; i<10; i++) {
        data_b[i] = i * 2;
    }
    int *shape_a = malloc(2 * sizeof(int));
    shape_a[0] = 2;
    shape_a[1] = 5;
    int *shape_b = malloc(2 * sizeof(int));
    shape_b[0] = 2;
    shape_b[1] = 5;

    int num_dims = 2;
    int size = 10;

    struct Tensor *a = tensor_from_data(data_a, shape_a, num_dims, size, true);
    struct Tensor *b = tensor_from_data(data_b, shape_b, num_dims, size, true);
    struct Tensor *c = tensor_add(a, b, true);

    assert(c->data != data_a);
    assert(c->data != data_b);
    for (int i=0; i<10; i++) {
        assert(c->data[i] == data_a[i] + data_b[i]);
    }
    assert(c->shape != shape_a);
    assert(c->shape != shape_b);
    for (int i=0; i<num_dims; i++) {
        assert(c->shape[i] == shape_a[i]);
        assert(c->shape[i] == shape_b[i]);
    }
    assert(c->num_dims == num_dims);
    assert(c->size == size);
    tensor_free(c);
}

void test_tensor_backward() {
    struct Tensor *a = make_tensor();
    struct Tensor *b = make_tensor();
    struct Tensor *c = tensor_add(a, b, true);
    tensor_backward(c);

    assert(a->backprop_config->gradient == 1);
    assert(b->backprop_config->gradient == 1);
    assert(c->backprop_config->gradient == 1);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

int main() {
    test_tensor_from_data();
    return 0;
}