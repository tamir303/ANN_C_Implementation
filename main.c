#include <stdio.h>

#include "Model/Network.h"

void main() {

	Config conf = 
	{
		.num_layers = 3,
		.attributes = {
			{2},
			{4, "relu"},
			{1, "sigmoid"}
		}
	};

	Network* ann = init_network(conf, "binary_crossentropy");

	Input x_trian = {
		.data = { {1, 2}, {3, 4 }, {5, 6} },
		.rows = 3,
		.cols = 2
	};

	Input y_train = {
		.data = {{1}, {0}, {1}},
		.rows = 3,
		.cols = 1
	};

	fit(ann, x_trian, y_train, 100);

	free_network(ann);
}