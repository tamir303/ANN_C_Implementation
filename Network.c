#include "Model/Network.h"
#include "Utils/CommonUtils.h"
#include "Utils/NetworkUtils.h"

Network* init_network(const Config config, const char* loss_name) {
	Network* ann;
	int curr_shape;
	int prev_shape;
	const char* actv;

	ann = (Network*) dynamicAllocation(sizeof(Network));
	ann->layers = (Layer*) dynamicAllocation(sizeof(Layer) * config.num_layers);
	LossInfo* info = getLossFunctionByName(loss_name);
	ann->function = info->function;
	ann->derivative = info->derivative;
	ann->num_layers = config.num_layers;
	ann->type = getTypeOfOutput(loss_name);

	#pragma omp parallel for
	for (int i = 0; i < ann->num_layers; i++) {
		int bool_notFirstLayer = i >= 1; // Check if not first layer
		curr_shape = config.attributes[i].shape;

		if (bool_notFirstLayer) {
			prev_shape = config.attributes[i - 1].shape;
			actv = config.attributes[i].activision;
			create_layer(
				&ann->layers[i],
				curr_shape,
				prev_shape,
				actv
			);
		}
		else
			create_layer(
				&ann->layers[i],
				curr_shape,
				0,
				NULL
			);
	}

	return ann;
}

void fit(Network* const ann, Input x_train, Input y_train, int epochs) {
	// Assert Network's Ouput Match Label's shape
	assert_n(y_train.cols == ann->layers[ann->num_layers - 1].num_neu, "Output shape doesn't match output layer");

	double loss = 0.0;
	double* prediction = NULL, * label = NULL;
	double* predictions = dynamicAllocation(sizeof(double) * y_train.rows);

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int inp_idx = 0; inp_idx < x_train.rows; inp_idx++) {
			prediction = forward(ann, x_train.data[inp_idx]);
			predictions[inp_idx] = classify_prediction(ann, prediction);
			backward(ann, y_train.data[inp_idx], DEFAULT_LEARNING_RATE);
		}

		loss = ann->function(predictions, y_train.data, y_train.rows);
		printf("Epoch %d: ==> Loss: %.2f\n", epoch, loss);
	}
}

void predict(Network* const ann, Input data) {
	assert_n(ann->layers[0].num_neu == data.cols, "Input shape doesn't match Input layer");

	Layer* output = &ann->layers[ann->num_layers - 1];
	int idx;

	for (int i = 0; i < data.rows; i++) {
		double* prediction = forward(ann, data.data[i]);
		double c_pred = classify_prediction(ann, prediction);

		if (ann->type != REG)
			printf("%d: %.2f", i, c_pred);

		printf("\n");
	}
}

void free_network(Network* ann) {
	for (int i = 0; i < ann->num_layers; i++)
		free_layer(&ann->layers[i]);
	free(ann->layers);
	free(ann);
}