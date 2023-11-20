#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "Model/Layer.h"
#include "Utils/CommonUtils.h"

Layer* create_layer(Layer* layer, int num_of_units, int num_of_weights, const char* activation_function) {
	ActivationInfo* act_info = getActivationFunctionByName(activation_function);
	assert_n(num_of_units > 0, "Number of neurons in a layer canno't be zero or negetive");

	if (act_info != NULL) {
		layer->actv = act_info->function;
		layer->dactv = act_info->derivative;
	}
	layer->num_neu = num_of_units;
	layer->neu = create_neurons(num_of_units, num_of_weights);

	return layer;
}

vector_t vector_layer(Layer* layer) {
	vector_t v = vector_init(layer->num_neu);

	#pragma omp parallel for
	for (int i = 0; i < layer->num_neu; i++)
		v[i] = layer->neu[i].z;

	return v;
}

void free_layer(Layer* layer) {
	for (int i = 0; i < layer->num_neu; i++)
		free_neuron(&layer->neu[i]);
	free(layer->neu);
}