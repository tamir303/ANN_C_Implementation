#include "Utils/CommonUtils.h"

void* dynamicAllocation(size_t size_t) {
	void* ptr = malloc(size_t);

	if (ptr == NULL) {
		fprintf(stderr, "Falied to allocate array");
		exit(1);
	}

	return ptr;
}

void free_2d_array(void** arr, size_t rows) {
	for (size_t i = 0; i < rows; i++)
		free(arr[i]);
	free(arr);
}

vector_t vector_init(size_t size) {
	vector_t v = (vector_t) calloc(size, sizeof(double));

	return v;
}

double vector_prod(vector_t v1, vector_t v2, int size) {
	double sum = 0.0;
	#pragma omp parallel for reduction(+ : sum)
	for (size_t i = 0; i < size; i++)
		sum += v1[i] * v2[i];

	return sum;
}

vector_t vector_mul(vector_t v1, vector_t v2, int size) {
	vector_t mul_v = (vector_t) dynamicAllocation(size * sizeof(double));
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++)
		mul_v[i] = v1[i] * v2[i];

	return mul_v;
}

void vector_free(vector_t v) {
	free(v);
}

//////////////////////////////////////////

int argmax(vector_t v, int size) {
	double max_val = v[0];
	int max_index = 0;

	#pragma omp parallel for reduction(max: max_val)
	for (int i = 0; i < size; i++) 
		if (v[i] > max_val) {
			max_val = v[i];
			max_index = i;
		}
	
	return max_index;
}