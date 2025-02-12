#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor.h"

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

    struct tensor *t = tensor_from_data(data, shape, num_dims, size, true);
    
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

int main() {
    test_tensor_from_data();
    return 0;
}