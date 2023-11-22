#pragma once

#include "../Model/Network.h"

#define MODEL_FILE_NAME "model.dat"

void saveModel(const Network* data);
void loadModel(const Network* data);

EOutputType getTypeOfOutput(const char* loss_name);

void forward(Network* ann, double* data);
void backward(Network* ann, double* label, double learning_rate);

double calculate_gradient_and_update_weights(Network* ann, int layer_idx, int n_idx, double error, double learning_rate);
double classify_prediction(Network* ann, double* prediction);