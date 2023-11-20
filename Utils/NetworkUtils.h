#pragma once

#include "../Model/Network.h"

EOutputType getTypeOfOutput(const char* loss_name);
double* forward(Network* ann, double* data);
void backward(Network* ann, double* label, double learning_rate);
double calculate_gradient_and_update_weights(Network* ann, int layer_idx, int n_idx, double error, double learning_rate);
double classify_prediction(Network* ann, double* prediction);