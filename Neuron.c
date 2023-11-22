#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "Model/Neuron.h"

void init_neuron(Neuron* n, int num_of_weights) {
	n->num_weights = num_of_weights;
	n->weights = vector_init(num_of_weights);
	n->z = 0.0;
	n->bias = 0;

	// Custom Glorot initialization
	const double SCALER = sqrt(6.f / num_of_weights);
	#pragma omp for
		for (int i = 0; i < num_of_weights; i++) {
			double rand_value = ( ((double) rand() / RAND_MAX) * 2 - 1 ) * SCALER;
			n->weights[i] = rand_value;
		}

	n->bias = (((double)rand() / RAND_MAX) * 2 - 1) * SCALER;
}

Neuron* create_neurons(int num_of_units, int num_of_weights) {
	Neuron* neurons = (Neuron*) dynamicAllocation(sizeof(Neuron) * num_of_units);
	#pragma omp parallel for
	for (int i = 0; i < num_of_units; i++)
		 init_neuron(&neurons[i], num_of_weights);

	return neurons;
}

void free_neuron(Neuron* neuron) {
	free(neuron->weights);
}