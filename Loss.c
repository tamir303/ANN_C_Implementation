#include <math.h>
#include <stdio.h>
#include <string.h>

#include "Model/Loss.h"

LossInfo* getLossFunctionByName(const char* func_name) {
	LossInfo lossFunctionsList[] = {
	{"mse", MSE, d_MSE},
	{"binary_crossentropy", BCE, d_BCE} };

	for (int i = 0; i < NUM_OF_LOSS; i++)
		if (strcmp(lossFunctionsList[i].name, func_name) == 0)
			return &lossFunctionsList[i];

	// Return NULL if function name wasn't found
	return NULL;
}

// #########  MSE  ######## ///
// MSE = (1 / N) * sum(Y_pred - Y_label)^2
double MSE(const double* predict, const double* label, int n) {
	double sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n; i++)
		sum += pow(predict[i] - label[i], 2);

	return sum / n;
}

double d_MSE(const double predict, const double label) {
	return (predict - label) * 2;
}

// #########  BCE  ######## ///
// BCE = - (1 / N) * sum(Y_label * log(Y_pred) + (1 - Y_label) * log(1 - Y_pred)
double BCE(const double* predict, const double* label, int n) {
	double sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n; i++)
		if (predict[i] == 1)
			sum += predict[i] != label[i];
		else
			sum += label[i] * log(predict[i]) + (1 - label[i]) * log(1 - predict[i]);

	return abs(sum) / (double) n;
}

double d_BCE(const double predict, const double label) {
	return predict - label;
}
