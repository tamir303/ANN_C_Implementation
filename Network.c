#include "Model/Network.h"
#include "Utils/CommonUtils.h"
#include "Utils/NetworkUtils.h"

Network* init_network(const Config config, const char* loss_name) {
	Network* ann;
	const char* actv;

	ann = (Network*) dynamicAllocation(sizeof(Network));
	ann->layers = (Layer*) dynamicAllocation(sizeof(Layer) * config.num_layers);
	LossInfo* info = getLossFunctionByName(loss_name);
	ann->function = info->function;
	ann->derivative = info->derivative;
	ann->num_layers = config.num_layers;
	ann->type = getTypeOfOutput(loss_name);

	create_layer(
		&ann->layers[0],
		config.input_layer,
		0,
		NULL
	);

	#pragma omp parallel for
	for (int i = 0; i < ann->num_layers - 1; i++) {
			create_layer(
				&ann->layers[i + 1],
				config.layers[i].shape,
				ann->layers[i].num_neu,
				config.layers[i].activision
			);
	}

	return ann;
}

void fit(Network* const ann, Input x_train, Input y_train, int epochs) {
	// Assert Network's Ouput Match Label's shape
	assert_n(y_train.cols == ann->layers[ann->num_layers - 1].num_neu, "Output shape doesn't match output layer");

	double loss = 0.0;
	double* label = NULL;
	double* predictions = dynamicAllocation(sizeof(double) * y_train.rows);
	Layer* output = &ann->layers[ann->num_layers - 1];

	normalize(x_train.data, x_train.rows, x_train.cols);
	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int inp_idx = 0; inp_idx < x_train.rows; inp_idx++) {
			forward(ann, x_train.data[inp_idx]);
			predictions[inp_idx] = classify_prediction(ann, vector_layer(output));
			backward(ann, y_train.data[inp_idx], DEFAULT_LEARNING_RATE);
		}

		loss = ann->function(predictions, y_train.data, y_train.rows);
		printf("Epoch %d: ==> Loss: %.2f\n", epoch + 1, loss);
	}
}

void predict(Network* const ann, Input data) {
	assert_n(ann->layers[0].num_neu == data.cols, "Input shape doesn't match Input layer");

	Layer* output = &ann->layers[ann->num_layers - 1];

	for (int i = 0; i < data.rows; i++) {
		forward(ann, data.data[i]);
		double c_pred = classify_prediction(ann, vector_layer(output));

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

void save_model(Network* ann) {
	saveModel(ann);
}

void load_model(Network* ann) {
	loadModel(ann);
}