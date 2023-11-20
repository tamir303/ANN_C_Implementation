#include <math.h>
#include <stdio.h>
#include <string.h>

#include "Model/Activations.h"

ActivationInfo* getActivationFunctionByName(const char* func_name) {
    if (func_name == NULL)
        return NULL;

    ActivationInfo activationFunctionsList[] = {
        {"sigmoid", Sigmoid, d_Sigmoid},
        {"relu", Relu, d_Relu}
    };

    for (int i = 0; i < NUM_OF_ACTIVATIONS; i++)
        if (strcmp(activationFunctionsList[i].name, func_name) == 0)
            return &activationFunctionsList[i];

    // Return NULL if function name wasn't found
    return NULL;
}

// #########  SIGMOID  ######## ///

double Sigmoid(const double z) {
    return 1.f / (1.f + exp(-z));
}

double d_Sigmoid(const double z) {
    double sigVal = Sigmoid(z);

    return sigVal * (1 - sigVal);
}

// #########  Leaky_RELU  ######## ///

double Relu(const double z) {
    return (z <= 0) ? 1e-2 * z : z;
}

double d_Relu(const double z) {
    return (z <= 0) ? 1e-2 : 1;
}