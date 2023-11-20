#include "Neuron.h"

#ifndef LAYER_H

#define LAYER_H

typedef struct layer_t {
    int num_neu;
    Neuron* neu;
    ActivationDerivativePointer dactv;
    ActivationFunctionPointer actv;
} Layer;

Layer* create_layer(Layer* layer, int num_of_units, int num_of_weights ,const char* activation_function);
vector_t vector_layer(Layer* layer);
void free_layer(Layer*);

#endif