#include "Utils/NetworkUtils.h"
#include "Utils/CommonUtils.h"

EOutputType getTypeOfOutput(const char* loss_name) {
	if (strcmp(loss_name, "mse") == 0)
		return REG;

	if (strcmp(loss_name, "binary_crossentropy") == 0)
		return BIN_CLASS;

	if (strcmp(loss_name, "softmax") == 0)
		return MUL_CLASS;

	return 0;
}

/**
 * @brief Performs the forward pass of the neural network to compute activations.
 *
 * @param ann Pointer to the neural network structure.
 * @param data Array representing the input data to the neural network.
 */
void forward(Network* ann, double* data) {
	// Get the number of layers in the neural network
	int num_layers = ann->num_layers;

	// Get the number of neurons in the first layer
	int first_layer_neu_num = ann->layers[0].num_neu;

	// Variables for activation function and the number of neurons in the current layer
	ActivationFunctionPointer actv;
	Layer* input_layer = &ann->layers[0];
	Layer* curr_layer;
	Neuron* curr_neuron;

	// Vector representing the outputs of the previous layer
	vector_t prev_layer_vect;

	// Insert input data into the input layer neurons
	#pragma omp parallel for
	for (int i = 0; i < first_layer_neu_num; i++)
		input_layer->neu[i].z = data[i];

	// Dataflow forward pass through all layers
	for (int i = 1; i < num_layers; i++) {
		curr_layer = &ann->layers[i];
		actv = curr_layer->actv;
		int curr_layer_neu_num = curr_layer->num_neu;
		prev_layer_vect = vector_layer(&ann->layers[i - 1]);

		// Compute the weighted sum and apply the activation function for each neuron
		#pragma omp parallel for
		for (int j = 0; j < curr_layer_neu_num; j++) {
			curr_neuron = &curr_layer->neu[j];
			double wx = vector_prod(prev_layer_vect, curr_neuron->weights, curr_neuron->num_weights);
			double z = actv(wx + curr_neuron->bias); // z = A(wx + b)
			curr_neuron->z = z;
		}
	}
}

/**
 * @brief Performs the backward pass of the neural network for training.
 *
 * This function calculates errors, gradients, and updates weights using backpropagation.
 *
 * @param ann Pointer to the neural network structure.
 * @param predicted Array representing the predicted output of the neural network.
 * @param label Array representing the true labels for the training data.
 * @param learning_rate Learning rate for weight updates during backpropagation.
 */
void backward(Network* ann,
	double* label,
	double learning_rate)
{
	LossDerivativePointer dloss = ann->derivative;
	Layer* layers = ann->layers;
	int num_layers = ann->num_layers;

	double** errors = (double**) dynamicAllocation(sizeof(double*) * num_layers);
	double** gradients = (double**) dynamicAllocation(sizeof(double*) * num_layers);

	#pragma omp parallel for
	for (int i = 0; i < num_layers; i++) {
		errors[i] = (double*) dynamicAllocation(sizeof(double) * layers[i].num_neu);
		gradients[i] = (double*) dynamicAllocation(sizeof(double) * layers[i].num_neu);
	}

	// Compute Output Layer Error and Gradients and Update Output Weights
	#pragma omp parallel for
	for (int n_idx = 0; n_idx < layers[num_layers - 1].num_neu; n_idx++) {
		double predicted = ann->layers[num_layers - 1].neu[n_idx].z;
		errors[num_layers - 1][n_idx] = predicted - label[n_idx];
		gradients[num_layers - 1][n_idx] = \
			calculate_gradient_and_update_weights(ann,
				num_layers - 1,
				n_idx,
				errors[num_layers - 1][n_idx],
				learning_rate);
	}

	// Backpropagate the Error and Compute Gradients for Hidden Layers
	for (int layer_idx = num_layers - 2; layer_idx > 0; layer_idx--) {
		#pragma omp parallel for
		for (int n_idx = 0; n_idx < layers[layer_idx].num_neu; n_idx++) {

			// Compute the total derivative for the current neuron in the hidden layer
			// Error[layer_idx][n_idx] = Sum( [layer_idx + 1]_w * [layer_idx + 1]_gradient ) * dactv(z)
			double total_derivative = 0.0;
			#pragma omp parallel for reduction(+: total_derivative)
			for (int w_idx = 0; w_idx < layers[layer_idx + 1].num_neu; w_idx++) {
				double w = layers[layer_idx + 1].neu[w_idx].weights[n_idx];
				double w_gradient = gradients[layer_idx + 1][w_idx];
				total_derivative += w_gradient * w;
			}

			// Set the error for the current neuron in the hidden layer
			errors[layer_idx][n_idx] = total_derivative;

			// Calculate the gradient and update weights using gradient descent
			gradients[layer_idx][n_idx] = calculate_gradient_and_update_weights(
				ann,
				layer_idx,
				n_idx,
				errors[layer_idx][n_idx],
				learning_rate);
		}
	}

	free_2d_array(gradients, num_layers);
	free_2d_array(errors, num_layers);
}

/**
 * @brief Calculates the gradient and updates weights for a neuron during backpropagation.
 *
 * @param ann Pointer to the neural network structure.
 * @param layer_idx Index of the current layer.
 * @param n_idx Index of the current neuron in the layer.
 * @param error Error for the current neuron.
 * @param learning_rate Learning rate for weight updates.
 *
 * @return The calculated gradient.
 */
double calculate_gradient_and_update_weights(Network* ann,
	int layer_idx,
	int n_idx,
	double error,
	double learning_rate) 
{
	Layer* layers = ann->layers;
	Layer* prev_layer = &layers[layer_idx - 1];
	Neuron* n = &layers[layer_idx].neu[n_idx];

	double z = layers[layer_idx].neu[n_idx].z;
	double dactv = layers[layer_idx].dactv(z);

	// Gradient[layer_idx][n_idx] = (dE / dO) * dactv(z)
	double gradient = error * dactv;

	// Update weights using gradient descent
	// New_W = Old_W - learning_rate * gradient * Input_idx
	#pragma omp parallel for
	for (int w_idx = 0; w_idx < n->num_weights; w_idx++) {
		n->weights[w_idx] -= learning_rate * gradient * prev_layer->neu[w_idx].z;
	}

	// Update bias using gradient descent
	n->bias -= learning_rate * gradient;

	return gradient;
}

double classify_prediction(Network* ann, double* prediction) {
	EOutputType type = ann->type;

	switch (type)
	{
	case BIN_CLASS:
		switch (ann->layers[ann->num_layers - 1].num_neu) // 1 or 2
		{
		case 1: // Output one neuron
			// Sigmoid actv, assume threshold of 0.5 for now
			return prediction[0] < 0.5 ? 0 : 1;

		case 2: // Output two neurons
			return prediction[0] > prediction[1] ? 0 : 1;
		}
		break;

	case MUL_CLASS:
		return argmax(prediction, ann->layers[ann->num_layers - 1].num_neu);
	}
}

void saveModel(const Network* ann) {
	FILE* file = fopen(MODEL_FILE_NAME, "wb");

	if (file != NULL) {
		fwrite(ann, sizeof(Network), 1, file);
		fclose(file);
	}
	else {
		fprintf(stderr, "Error, Failed To Save File");
	}
}

void loadModel(const Network* ann) {
	FILE* file = fopen(MODEL_FILE_NAME, "rb");

	if (file != NULL) {
		fread(ann, sizeof(Network), 1, file);
		fclose(file);
	}
	else {
		fprintf(stderr, "Error, Failed To Save File");
	}
}