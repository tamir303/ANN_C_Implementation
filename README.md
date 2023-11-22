# Neural Network C Implementation

![Neural Network C Implementation Logo](logo.png)

Neural Network C Implementation is a C library for building and training artificial neural networks. This library is designed to provide a simple yet flexible framework for constructing and training neural networks in C.

## Features

- **Create Neural Network Models:** Build neural network models with customizable architectures suited to your specific needs.

- **Layer Customization:** Define layers, including input, hidden, and output layers, and customize their properties.

- **Activation Functions:** Specify activation functions for layers to introduce non-linearity and adaptability to your models.

- **Loss Functions and Optimizers:** Train models using various loss functions and optimizers to fine-tune your network's performance.

- **Easy Integration:** Designed to be easy to integrate into C projects, making it ideal for embedded systems and custom solutions.

## Installation

To get started with Neural Network C Implementation, follow these steps:

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/yourusername/Neural-Network-C-Implementation.git

## Usage

```c
#include "Model/Network.h"

void main() {
	
	// Create Layers configurations for Network
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
	
	// Initalize Network
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

	// Train Network 
	fit(ann, x_train, y_train, 10);
	
	// Save model to be loaded later
	save_model(ann);

	// Free Network
	free_network(ann);
}