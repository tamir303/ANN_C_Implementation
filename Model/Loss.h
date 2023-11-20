#pragma once

#define NUM_OF_LOSS 2

// Mean Squeard Error
double MSE(const double* predict, const double* label, int n);
double d_MSE(const double predict, const double label);

// Binary CrossEntropy
double BCE(const double* predict, const double* label, int n);
double d_BCE(const double predict, const double label);

typedef double (*LossFunctionPointer)(const double* predict, const double* label, int n);
typedef double (*LossDerivativePointer)(const double predict, const double label);

typedef struct lossInfo_t
{
    const char* name;
    LossFunctionPointer function;
    LossDerivativePointer derivative;
} LossInfo;

LossInfo* getLossFunctionByName(const char* func_name);