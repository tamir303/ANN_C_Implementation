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

int main() {
    // Create a new neural network
    Network* network = initEmptyNetwork();

    // Define input layer
    Layer* input_layer = Input(3);

    // Define hidden layer with a specified activation function
    Layer* hidden_layer = Dense(6, "relu");

    // Define output layer with a different activation function
    Layer* output_layer = Output(2, "sigmoid");

    // Add layers to the network
    add(network, 3, input_layer, hidden_layer, output_layer);

    // Compile the model with a loss function
    compile(network, "mse");

    // Train the model and make predictions
    // ...

    // Clean up memory
    destroyNetwork(network);

    return 0;
}
