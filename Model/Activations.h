#pragma once

#define NUM_OF_ACTIVATIONS 2

double Sigmoid(const double z);
double d_Sigmoid(const double z);

double Relu(const double z);
double d_Relu(const double z);


typedef double (*ActivationFunctionPointer)(const double z);
typedef double (*ActivationDerivativePointer)(const double z);

typedef struct activationInfo_t
{
    const char* name;
    ActivationFunctionPointer function;
    ActivationDerivativePointer derivative;
} ActivationInfo;

ActivationInfo* getActivationFunctionByName(const char* func_name);