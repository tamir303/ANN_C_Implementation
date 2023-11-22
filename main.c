#include "Model/Network.h"

void main() {

	Config conf =
	{
		.num_layers = 4,
		.input_layer = 2,
		.layers = {
			{32, "relu"},
			{64, "relu"},
			{1, "sigmoid"}
		}
	};

	Network* ann = init_network(conf, "binary_crossentropy");

	Input x_train = {
		.data = { {1, 2}, {3, 4 }, {5, 6}, {5, 5 }, {5, 4}, {4, 5}, {9, 10} },
		.rows = 6,
		.cols = 2
	};

	Input y_train = {
		.data = {{0}, {0}, {1}, {1}, {0}, {0}, {1} },
		.rows = 6,
		.cols = 1
	};

	fit(ann, x_train, y_train, 10);
	
	save_model(ann);
	free_network(ann);
}