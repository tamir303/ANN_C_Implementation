#include "Activations.h"
#include "../Utils/CommonUtils.h"

#ifndef NEURON_H
#define NEURON_H

typedef struct neuron_t {
	int num_weights;
	vector_t weights;
	double bias;
	double z;
} Neuron;

void init_neuron(Neuron* n, int num_of_weights);
Neuron* create_neurons(int num_of_units, int num_of_weights);
void free_neuron(Neuron* n);

#endif