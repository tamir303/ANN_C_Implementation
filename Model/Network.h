#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include "Loss.h"
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_LEARNING_RATE 1e-1
#define MAX_DIM_SIZE 200
#define MAX_LAYERS 50
#define MAX_ROWS 200
#define MAX_COLS 30

typedef struct input_t { 
	double data[MAX_ROWS][MAX_COLS];
	int rows;
	int cols; 
} Input;

typedef struct config_attr_t {
	int shape;
	const char* activision;
} Config_att;

/**
 * @struct Config
 * @brief Holds configuration information for a neural network. 
 * @brief Example: { 4, { {5}, {10, "relu"}, {30, "relu"}, {2, "sigmoid"} } };
 * @param int [num_layers] The actual number of layers in use
 * @param Config_att [attributes] represents the configuration attributes for a neural network layer
 */
typedef struct config_t {
	int num_layers;
	int input_layer;
	Config_att layers[MAX_LAYERS];
} Config;

typedef enum {
	REG,
	BIN_CLASS,
	MUL_CLASS
} EOutputType;

typedef struct Network {
	Layer* layers;
	int num_layers;
	LossFunctionPointer function;
	LossDerivativePointer derivative;
	EOutputType type;
} Network;

Network* init_network(const Config config, const char* loss_name);
void fit(Network* const ann, Input x_train, Input y_train, int epochs);
void predict(Network* const network, Input data);
void free_network(Network*);
void save_model(Network*);
void load_model(Network*);

#endif 
